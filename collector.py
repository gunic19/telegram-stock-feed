"""Collect short public Telegram post excerpts and optionally process shortlisted text with Gemini."""
import datetime as dt
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
import ai_classifier

CHANNELS = ['stockpro_online','motilaloswalofficial','GHANSHYAMTECH','Equiideas','STOCKGAINERSS','eagleeyesmarketanalysis','stockmarketadviser91','stockboxtradinggg']
OUTPUT = Path('feed.json')
MAX_EVENTS = 1600
MAX_DAYS = 30
MAX_PAGES = 4
USER_AGENT = {'User-Agent': 'Mozilla/5.0 (compatible; PublicChannelReader/1.0)'}
IGNORE_WORDS = {'THE','THIS','NOW','CALL','CALLS','PUT','PUTS','OPTION','OPTIONS','ABOVE','BELOW','AT','ON','IN','A','STOCK','STOCKS','INDEX','NIFTY','BANKNIFTY','SENSEX','FINNIFTY','CMP','TGT','TARGET','LEVEL','NEAR','ONLY','SIGNAL','DIP','DIPS','ZONE','FUTURE','FUTURES','BUY','SELL','FOR','SL'}
NUM = r'(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)'
ACTION = re.compile(r'\b(BUY|SELL)\s+(?:#|\$)?([A-Z][A-Z0-9&.\-]{1,24})\b')
RELEVANT = re.compile(r'\b(?:BUY|SELL|WATCH|TARGET|TGT|STOP\s*LOSS|\bSL\b|BREAKOUT|RECOMMENDATION|HOLD|EXIT|BOOK\s+PROFIT)\b', re.I)
EXCLUDE_ADS = re.compile(r'\b(?:JOIN\s+(?:OUR\s+)?(?:PREMIUM|PRIME|PAID)|ENROLL\s+NOW|ADMISSIONS?\s+(?:ARE\s+)?OPEN|REGISTRATIONS?\s+ARE\s+OPEN)\b',re.I)
OPTIONS = re.compile(r'\b(?:OPTION(?:S)?|FUTURES?|NIFTY|BANKNIFTY|SENSEX|FINNIFTY|CALL\s+OPTION|PUT\s+OPTION|\d{4,6}\s*(?:CE|PE))\b',re.I)

def redact(text):
    text=re.sub(r'https?://\S+','[link]',text)
    text=re.sub(r'\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b','[email]',text)
    text=re.sub(r'(?<!\w)(?:\+91[\s-]*)?[6-9]\d{4}[\s-]?\d{5}(?!\w)','[phone]',text)
    return re.sub(r'[\t\r ]+',' ',text).strip()[:700]

def extract_number(pattern,text):
    hit=re.search(pattern+r'\s*[:=@₹-]?\s*'+NUM,text,re.I)
    if not hit:return None
    # Percentages and numbered labels like 'TARGET 1: 450' are not prices.
    # Ambiguous ranges/numbered targets stay in the original excerpt for review.
    tail=text[hit.end():]
    if re.match(r'\s*(?:%|/\s*\d|[-–]\s*\d|:\s*\d)',tail):return None
    if hit.group(1)=='1' and re.match(r'\s+\d',tail):return None
    try:
        n=float(hit.group(1).replace(',',''))
        return n if 0<n<1e8 else None
    except ValueError:return None

def classify(text,media=False):
    if not text:return {'category':'review','reason':'Media or no readable text; open the original post.'} if media else None
    if EXCLUDE_ADS.search(text):return None
    if not RELEVANT.search(text):
        if ai_classifier.NEWS.search(text):return {'category':'news','reason':'Possible company news; AI/name and original require verification.'}
        return {'category':'review','reason':'Media attached; image contents not read. Open original.'} if media else None
    # The regex intentionally accepts only an explicit BUY/SELL followed by
    # one uppercase token; every result is a candidate, not a verified signal.
    actions=list(ACTION.finditer(text))
    result={'category':'review','reason':'Ambiguous or update/news; verify original post.'}
    if len(actions)==1 and not OPTIONS.search(text):
        hit=actions[0];token=hit.group(2)
        if token not in IGNORE_WORDS:
            result={'category':'call','side':hit.group(1),'symbol_candidate':token,
                    'reason':'Automatically detected, not verified. Confirm symbol, entry and context.'}
            result['entry_as_posted']=extract_number(r'\bCMP\b',text)
            result['first_target']=extract_number(r'\b(?:TGT|TARGET)\b',text)
            result['stop_loss']=extract_number(r'\b(?:SL|STOP\s*LOSS)\b',text)
            return result
    if re.search(r'\b(?:WATCHLIST|WATCH)\b',text,re.I) and not actions:
        result={'category':'watch','reason':'Watch wording detected; confirm symbol and context.'}
    return result

def parse_page(channel,html):
    soup=BeautifulSoup(html,'html.parser');posts=[]
    for m in soup.select('.tgme_widget_message[data-post]'):
        post=str(m.get('data-post',''));prefix,sep,id_=post.rpartition('/')
        if not sep or prefix.lower()!=channel.lower() or not id_.isdigit():continue
        tm=m.select_one('time[datetime]');box=m.select_one('.tgme_widget_message_text')
        if tm is None:continue
        try: posted=dt.datetime.fromisoformat(tm.get('datetime').replace('Z','+00:00')).astimezone(dt.timezone.utc).isoformat()
        except Exception:continue
        text=redact(box.get_text(' ',strip=True) if box else '')
        has_media=bool(m.select_one('.tgme_widget_message_photo_wrap,.tgme_widget_message_video_player,.tgme_widget_message_document_wrap,.tgme_widget_message_video_wrap'))
        parsed=classify(text,has_media)
        if parsed is None:continue
        digest=hashlib.sha256((text+'|'+str(has_media)).encode('utf-8')).hexdigest()[:16]
        posts.append({'channel':channel,'message_id':int(id_),'posted_at':posted,'excerpt':text,'has_media':has_media,'version':digest,**parsed})
    all_ids=[int(x.get('data-post').rpartition('/')[2]) for x in soup.select('.tgme_widget_message[data-post]') if x.get('data-post','').rpartition('/')[2].isdigit()]
    return posts,all_ids

def fetch_channel(channel,previous):
    out=[]; before=None; reached=False; pages=0;max_seen=0
    try:
        for _ in range(MAX_PAGES):
            url='https://t.me/s/'+channel
            r=requests.get(url,params={'before':before} if before else None,headers=USER_AGENT,timeout=(6,15))
            if r.status_code!=200:raise RuntimeError('HTTP '+str(r.status_code))
            if len(r.content)>2_500_000:raise RuntimeError('preview too large')
            messages,ids=parse_page(channel,r.text);pages+=1
            if not ids:
                if pages==1:raise RuntimeError('No public post preview available')
                reached=True;break
            max_seen=max(max_seen,max(ids));out.extend(messages)
            if not previous or min(ids)<=previous:reached=True;break
            new_before=min(ids)
            if new_before==before:break
            before=new_before
        return {'state':'ok' if reached else 'partial','message':('Latest public preview checked' if reached else 'Backfill limit reached; some posts may be missing'),'pages':pages,'newest_id':max_seen,'observed':out}
    except Exception as ex:
        return {'state':'error','message':str(ex)[:140],'pages':pages,'newest_id':max_seen,'observed':out}

def run():
    now=dt.datetime.now(dt.timezone.utc);iso=now.isoformat()
    try:
        old=json.loads(OUTPUT.read_text(encoding='utf-8')) if OUTPUT.exists() else {}
        if old.get('schema')!=1: old={}
    except (OSError,ValueError): old={}
    states=old.get('sources',{}) if isinstance(old.get('sources',{}),dict) else {}
    work={ch:int(states.get(ch,{}).get('newest_id',0) or 0) for ch in CHANNELS}
    with ThreadPoolExecutor(max_workers=4) as pool: results=dict(zip(CHANNELS,pool.map(lambda ch:fetch_channel(ch,work[ch]),CHANNELS)))
    existing=old.get('events',[]) if isinstance(old.get('events',[]),list) else []
    events={(x.get('channel'),x.get('message_id'),x.get('version')):x for x in existing if isinstance(x,dict)}
    for channel,r in results.items():
        prev=states.get(channel,{}) if isinstance(states.get(channel,{}),dict) else {}
        for post in r.pop('observed'):
            key=(post['channel'],post['message_id'],post['version'])
            if key not in events:post['observed_at']=iso;events[key]=post
        # A bounded backfill or an interrupted later page can leave permanent gaps.
        # Keep the warning sticky rather than implying the missing posts were recovered.
        gap = bool(prev.get('gap_detected')) or r['state']=='partial' or (r['state']=='error' and r['pages']>0)
        states[channel]={**r,'gap_detected':gap,'checked_at':iso,'last_success':iso if r['state'] in ('ok','partial') else prev.get('last_success'), 'newest_id':max(r['newest_id'],work[channel])}
    cutoff=now-dt.timedelta(days=MAX_DAYS);kept=[]
    for x in events.values():
        try:
            when=dt.datetime.fromisoformat(x['posted_at'].replace('Z','+00:00'))
            if when.tzinfo is None:continue
            if cutoff<=when<=now+dt.timedelta(days=1):kept.append(x)
        except (ValueError,TypeError,KeyError):continue
    kept.sort(key=lambda x:(x['posted_at'],x.get('observed_at','')),reverse=True)
    kept=kept[:MAX_EVENTS]
    ai_state=ai_classifier.process(kept,old.get('ai_status',{}),now)
    data={'schema':1,'updated_at':iso,'retention':'At most 30 days / 1600 records. Initial run collects only the latest public preview.','sources':{x:states[x] for x in CHANNELS},'ai_status':ai_state,'events':kept}
    tmp=OUTPUT.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,OUTPUT)
    print('Collected',len(data['events']),'candidate/review versions;',', '.join(k+':'+states[k]['state'] for k in CHANNELS))
    if all(v['state']=='error' for v in states.values()):print('All sources failed; prior records retained.',file=sys.stderr)
if __name__=='__main__':run()
