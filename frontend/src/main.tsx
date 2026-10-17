import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import './style.css';

const API='http://127.0.0.1:8000';
type Model={id:string;name:string;provider:string;configured:boolean;quality:number;speed:number;context:number;strengths:string[];free:boolean};
type Result={answer:string;selected_model:string;provider:string;complexity:string;latency_ms:number;usage:Record<string,number>;decision:{ranked:any[];selected_reason:string[];fallbacks_attempted:any[]}};

function App(){
 const [models,setModels]=useState<Model[]>([]); const [metrics,setMetrics]=useState<any>({});
 const [prompt,setPrompt]=useState('Design a scalable distributed cache for a high-traffic e-commerce system. Explain the architecture, consistency trade-offs, eviction policy, sharding, failure handling, and provide a Python implementation outline.');
 const [quality,setQuality]=useState(.85); const [latency,setLatency]=useState(15000); const [privacy,setPrivacy]=useState('standard'); const [loading,setLoading]=useState(false); const [result,setResult]=useState<Result|null>(null); const [error,setError]=useState('');
 const refresh=()=>{fetch(API+'/api/models').then(r=>r.json()).then(setModels).catch(()=>{}); fetch(API+'/api/metrics').then(r=>r.json()).then(setMetrics).catch(()=>{});};
 useEffect(()=>{refresh(); const t=setInterval(refresh,5000); return()=>clearInterval(t)},[]);
 const route=async()=>{setLoading(true);setError('');setResult(null);try{const r=await fetch(API+'/api/route',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({prompt,quality_target:quality,latency_budget_ms:latency,privacy,max_tokens:1200})}); const d=await r.json(); if(!r.ok) throw new Error(typeof d.detail==='string'?d.detail:JSON.stringify(d.detail)); setResult(d); refresh();}catch(e:any){setError(e.message)}finally{setLoading(false)}};
 return <div className="app"><header><div><div className="eyebrow">INTELLIGENT INFERENCE GATEWAY</div><h1>AI Model Router</h1><p>One API. Five real free-tier models. Every request is routed by quality, complexity and latency.</p></div><div className="status"><span className="dot"/>LIVE ROUTING</div></header>
 <section className="hero"><div className="heroText"><span className="pill">NO MOCK PROVIDER</span><h2>Let the router decide.</h2><p>Submit a real prompt and inspect the complete decision trace: candidate scores, selected model, latency, usage and fallback behavior.</p></div><div className="stats"><div><b>{models.filter(m=>m.configured).length}/{models.length||5}</b><span>models configured</span></div><div><b>{metrics.success_rate||0}%</b><span>success rate</span></div><div><b>{Math.round(metrics.avg_latency_ms||0)} ms</b><span>avg latency</span></div></div></section>
 <main><section className="panel playground"><div className="panelHead"><div><span className="eyebrow">ROUTING PLAYGROUND</span><h3>Request policy</h3></div><button onClick={route} disabled={loading}>{loading?'Routing…':'Route request →'}</button></div>
 <textarea value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="Ask anything…"/>
 <div className="controls"><label>Quality target <strong>{quality.toFixed(2)}</strong><input type="range" min=".5" max="1" step=".01" value={quality} onChange={e=>setQuality(+e.target.value)}/></label><label>Latency budget <strong>{latency/1000}s</strong><input type="range" min="1000" max="60000" step="1000" value={latency} onChange={e=>setLatency(+e.target.value)}/></label><label>Privacy<select value={privacy} onChange={e=>setPrivacy(e.target.value)}><option value="standard">Standard cloud</option><option value="sensitive">Sensitive (blocks free cloud)</option></select></label></div>
 {error&&<div className="error">{error}</div>}
 {result&&<div className="answer"><div className="answerMeta"><span className="selected">SELECTED · {result.selected_model}</span><span>{result.provider}</span><span>{result.complexity.toUpperCase()}</span><span>{result.latency_ms} ms</span></div><pre>{result.answer}</pre></div>}
 </section>
 <aside><section className="panel"><div className="panelHead"><div><span className="eyebrow">MODEL POOL</span><h3>Real endpoints</h3></div></div>{models.map(m=><div className="model" key={m.id}><div className="modelTop"><div><b>{m.name}</b><small>{m.provider}</small></div><span className={m.configured?'ready':'off'}>{m.configured?'READY':'KEY MISSING'}</span></div><div className="bars"><span style={{width:`${m.quality*100}%`}}/><span style={{width:`${m.speed*100}%`}}/></div><small>quality {m.quality.toFixed(2)} · speed {m.speed.toFixed(2)} · {Math.round(m.context/1000)}K ctx</small></div>)}</section></aside></main>
 {result&&<section className="panel trace"><div className="panelHead"><div><span className="eyebrow">DECISION TRACE</span><h3>Why this model won</h3></div></div><div className="ranked">{result.decision.ranked.map((x:any,i:number)=><div className={i===0?'rank winner':'rank'} key={x.model}><div><b>#{i+1} {x.name}</b><small>{x.reasons.join(' · ')}</small></div><strong>{x.score}</strong></div>)}</div></section>}
 <footer>AI Model Router · FastAPI + React + TypeScript · Real API inference only</footer></div>
}
createRoot(document.getElementById('root')!).render(<App/>);
