(function(){
'use strict';
var RP=30000,RS=30000,RH=60000,RY=60000,CI=1000;
var charts={},historyData={},stateData={},sortCol='timestamp',sortDir='desc',currentVersion='v209';

document.addEventListener('DOMContentLoaded',function(){
    initTabs();initClock();initSorting();initVersionToggle();fetchAll();
    setInterval(fetchPrices,RP);setInterval(fetchStatus,RS);
    setInterval(fetchHistory,RH);setInterval(fetchSystem,RY);
    var rb=document.getElementById('refresh-log');if(rb)rb.addEventListener('click',fetchHistory);
    var f=document.getElementById('log-asset-filter');if(f)f.addEventListener('change',renderTradeLog);
    setAutoReload(true);
});

function fetchAll(){fetchStatus();fetchPrices();fetchHistory();fetchSystem();}

/* ═══ RELOJ ═══ */
function initClock(){updateClock();setInterval(updateClock,CI);}
function updateClock(){
    var now=new Date();
    var t=now.toLocaleTimeString('es-CU',{hour:'2-digit',minute:'2-digit',second:'2-digit',timeZone:'America/Havana'});
    var d=now.toLocaleDateString('es-CU',{weekday:'long',day:'numeric',month:'long',year:'numeric',timeZone:'America/Havana'});
    el('live-clock',t);el('live-date',d);el('last-update',t);
}

/* ═══ VERSION TOGGLE ═══ */
function initVersionToggle(){
    document.querySelectorAll('.version-btn').forEach(function(btn){
        btn.addEventListener('click',function(){
            var ver=btn.dataset.version;
            document.querySelectorAll('.version-btn').forEach(function(b){b.classList.remove('active');});
            btn.classList.add('active');
            currentVersion=ver;
            el('version-display',ver==='v209'?'V20.9':'V21.0');
            document.querySelectorAll('.v21-only').forEach(function(el){el.style.display=ver==='v21'?'':'none';});
            fetchStatus();fetchHistory();
            if(ver==='v21')fetchComparison();
        });
    });
}

/* ═══ TABS ═══ */
function initTabs(){
    document.querySelectorAll('.tab-btn').forEach(function(btn){
        btn.addEventListener('click',function(){
            var tab=btn.dataset.tab;
            document.querySelectorAll('.tab-btn').forEach(function(b){b.classList.remove('active');});
            document.querySelectorAll('.tab-pane').forEach(function(p){p.classList.remove('active');});
            btn.classList.add('active');
            var pane=document.getElementById(tab);if(pane)pane.classList.add('active');
            if(tab==='system')fetchSystem();
            if(tab==='trades')fetchHistory();
            if(tab==='charts'){fetchHistory();setTimeout(function(){for(var k in charts)if(charts[k])charts[k].resize();},100);}
        });
    });
}

/* ═══ STATUS (state JSON) ═══ */
function fetchStatus(){
    var endpoint=currentVersion==='v209'?'/api/status':'/api/v21/status';
    fetch(endpoint).then(function(r){if(!r.ok)throw new Error();return r.json();})
    .then(function(d){if(d.error)throw new Error(d.error);stateData[currentVersion]=d;renderStatus(d);renderAssetState(d);updateBadge('activo');})
    .catch(function(){updateBadge('error');});
}

function renderStatus(data){
    var p=data.portfolio||{};
    var tb=p.total_balance||0,pk=p.portfolio_peak||30000,dd=p.portfolio_dd||0;
    var ib=30000,pnl=((tb-ib)/ib*100);
    anim('portfolio-balance','$'+fN(tb,2));
    var pe=document.getElementById('portfolio-pnl');
    if(pe){pe.textContent=(pnl>=0?'+':'')+pnl.toFixed(2)+'%';pe.className='value '+(pnl>=0?'positive':'negative');}
    anim('portfolio-peak','$'+fN(pk,2));
    anim('portfolio-dd',(dd*100).toFixed(2)+'%');
    var assets=data.assets||{},bars=0;
    for(var k in assets)if(assets[k]&&assets[k].bar_count)bars+=assets[k].bar_count;
    anim('step-count',bars.toString());
    var lu=data.last_update||'';
    if(lu){var dd2=new Date(lu);el('state-update',dd2.toLocaleString('es-CU',{timeZone:'America/Havana'}));}
}

function renderAssetState(data){
    var assets=data.assets||{};
    var c=document.getElementById('asset-state-cards');if(!c)return;
    var icons={'ETH/USDT':'⟠','BTC/USDT':'₿','SOL/USDT':'◎'};
    var colors={'ETH/USDT':'#627eea','BTC/USDT':'#f7931a','SOL/USDT':'#00ffa3'};
    var h='';
    for(var sym in assets){
        var info=assets[sym],ic=icons[sym]||'',cl=colors[sym]||'#00a8ff';
        var lp=info.ic_history&&info.ic_history.length>0?info.ic_history[info.ic_history.length-1].proba:0;
        var posPct=(info.prev_position*100).toFixed(1);
        var ddPct=(info.current_dd*100).toFixed(2);
        var driftTxt=info.drift_counter>0?'Alerta: '+info.drift_counter:'OK (0)';
        var driftColor=info.drift_counter>0?'var(--orange)':'var(--green)';
        var posColor='var(--green)';
        if(info.prev_position>0.4)posColor='var(--orange)';
        if(info.prev_position>0.45)posColor='var(--red)';
        var signal=lp>0.5?'Alta Vol':'Normal';
        var sigColor=lp>0.5?'var(--red)':'var(--green)';
        var pnlUsd=info.virtual_balance-10000;
        var pnlPct=((info.virtual_balance-10000)/10000*100).toFixed(2);
        var pnlColor=pnlUsd>=0?'var(--green)':'var(--red)';

        // V21 edge state for ETH
        var edgeHtml='';
        if(currentVersion==='v21'&&sym==='ETH/USDT'){
            var edgeState=info.last_edge_state||'N/A';
            var edgeProba=info.last_edge_proba||0;
            var edgeColor='var(--text-secondary)';
            if(edgeState==='EDGE_ON')edgeColor='var(--green)';
            if(edgeState==='EDGE_OFF')edgeColor='var(--red)';
            edgeHtml='<div class="asset-state-item"><span class="asset-state-label">Edge State</span><span class="asset-state-value" style="color:'+edgeColor+'">'+edgeState+'</span></div>';
            edgeHtml+='<div class="asset-state-item"><span class="asset-state-label">Edge Prob</span><span class="asset-state-value">'+edgeProba.toFixed(4)+'</span></div>';
        }

        h+='<div class="asset-state-card" style="border-left:3px solid '+cl+'">'
          +'<div class="asset-state-header"><span style="color:'+cl+';font-size:18px">'+ic+'</span><strong>'+sym+'</strong>'
          +'<span class="badge badge-success" style="font-size:10px">Barra '+( info.bar_count||0)+'</span></div>'
          +'<div class="asset-state-grid">'
          +'<div class="asset-state-item"><span class="asset-state-label">Balance Virtual</span><span class="asset-state-value">$'+fN(info.virtual_balance,2)+'</span></div>'
          +'<div class="asset-state-item"><span class="asset-state-label">P&L</span><span class="asset-state-value" style="color:'+pnlColor+'">'+(pnlUsd>=0?'+':'')+fN(pnlUsd,2)+' ('+pnlPct+'%)</span></div>'
          +'<div class="asset-state-item"><span class="asset-state-label">Posicion Actual</span><span class="asset-state-value" style="color:'+posColor+'">'+posPct+'%</span></div>'
          +'<div class="asset-state-item"><span class="asset-state-label">Prob. Alta Vol</span><span class="asset-state-value">'+(lp*100).toFixed(1)+'%</span></div>'
          +'<div class="asset-state-item"><span class="asset-state-label">Ultimo Cierre</span><span class="asset-state-value">$'+fN(info.prev_close,2)+'</span></div>'
          +'<div class="asset-state-item"><span class="asset-state-label">Drawdown</span><span class="asset-state-value" style="color:'+(parseFloat(ddPct)<-1?'var(--red)':'var(--text-secondary)')+'">'+ddPct+'%</span></div>'
          +'<div class="asset-state-item"><span class="asset-state-label">Drift Counter</span><span class="asset-state-value" style="color:'+driftColor+'">'+driftTxt+'</span></div>'
          +'<div class="asset-state-item"><span class="asset-state-label">Senal</span><span class="asset-state-value" style="color:'+sigColor+'">'+signal+'</span></div>'
          +edgeHtml
          +'</div>'
          +'<div class="position-bar-container"><div class="position-bar-label">Exposicion: '+posPct+'% de 50% max</div>'
          +'<div class="position-bar-track"><div class="position-bar-fill" style="width:'+Math.min(info.prev_position/0.5*100,100)+'%;background:'+cl+'"></div></div></div></div>';
    }
    c.innerHTML=h;
}

/* ═══ COMPARISON ═══ */
function fetchComparison(){
    fetch('/api/status').then(function(r){if(!r.ok)throw new Error();return r.json();})
    .then(function(v209){stateData['v209']=v209;return fetch('/api/v21/status');})
    .then(function(r){if(!r.ok)throw new Error();return r.json();})
    .then(function(v21){stateData['v21']=v21;renderComparison();})
    .catch(function(e){console.warn('Comparison:',e);});
}

function renderComparison(){
    var v209=stateData['v209']||{},v21=stateData['v21']||{};
    var cc=document.getElementById('comparison-card');if(!cc)return;
    if(!v209.portfolio||!v21.portfolio){cc.style.display='none';return;}
    cc.style.display='block';

    var v209Bal=v209.portfolio.total_balance||0,v21Bal=v21.portfolio.total_balance||0;
    var v209DD=v209.portfolio.portfolio_dd||0,v21DD=v21.portfolio.portfolio_dd||0;
    var v209EthPos=(v209.assets&&v209.assets['ETH/USDT']?v209.assets['ETH/USDT'].prev_position:0);
    var v21EthPos=(v21.assets&&v21.assets['ETH/USDT']?v21.assets['ETH/USDT'].prev_position:0);

    el('comp-v209-balance','$'+fN(v209Bal,2));
    el('comp-v21-balance','$'+fN(v21Bal,2));
    el('comp-v209-dd',(v209DD*100).toFixed(2)+'%');
    el('comp-v21-dd',(v21DD*100).toFixed(2)+'%');
    el('comp-v209-eth-pos',(v209EthPos*100).toFixed(1)+'%');
    el('comp-v21-eth-pos',(v21EthPos*100).toFixed(1)+'%');
}

/* ═══ PRICES ═══ */
function fetchPrices(){
    fetch('/api/prices').then(function(r){if(!r.ok)throw new Error();return r.json();})
    .then(function(d){renderCards(d);renderAllocChart(d);})
    .catch(function(e){console.warn('Prices:',e);});
}

function renderCards(prices){
    var c=document.getElementById('asset-cards');if(!c)return;
    var icons={'ETH/USDT':'⟠','BTC/USDT':'₿','SOL/USDT':'◎'};
    var colors={'ETH/USDT':'#627eea','BTC/USDT':'#f7931a','SOL/USDT':'#00ffa3'};
    var h='';
    for(var s in prices){
        var i=prices[s],cc=i.change_pct>=0?'positive':'negative',cs=i.change_pct>=0?'+':'';
        var ic=icons[s]||'',cl=colors[s]||'#00a8ff';
        var range=i.high24h-i.low24h,rangePct=i.low24h>0?((range/i.low24h)*100).toFixed(2):'0';
        var curPos=i.price-i.low24h,rangeFill=range>0?((curPos/range)*100).toFixed(0):'50';
        h+='<div class="asset-card" style="border-top:2px solid '+cl+'">'
          +'<div class="asset-header"><span class="asset-icon" style="color:'+cl+'">'+ic+'</span><span class="asset-name">'+s+'</span><span class="asset-change '+cc+'">'+cs+i.change_pct.toFixed(2)+'%</span></div>'
          +'<div class="asset-price">$'+fN(i.price,2)+'</div>'
          +'<div class="asset-details">'
          +'<div><span class="asset-detail-label">Maximo 24h</span><span class="asset-detail-value">$'+fN(i.high24h,2)+'</span></div>'
          +'<div><span class="asset-detail-label">Minimo 24h</span><span class="asset-detail-value">$'+fN(i.low24h,2)+'</span></div>'
          +'<div><span class="asset-detail-label">Volumen 24h</span><span class="asset-detail-value">$'+fC(i.vol24h)+'</span></div>'
          +'<div><span class="asset-detail-label">Rango 24h</span><span class="asset-detail-value">'+rangePct+'%</span></div></div>'
          +'<div class="price-range-bar"><div class="price-range-track"><div class="price-range-fill" style="width:'+rangeFill+'%;background:'+cl+'"></div></div>'
          +'<div class="price-range-labels"><span>$'+fN(i.low24h,0)+'</span><span>$'+fN(i.high24h,0)+'</span></div></div></div>';
    }
    c.innerHTML=h;
}

function renderAllocChart(prices){
    var cv=document.getElementById('allocation-chart');if(!cv)return;
    var assets=stateData[currentVersion]&&stateData[currentVersion].assets?stateData[currentVersion].assets:{};
    var lb=[],vl=[],bg=['#627eea','#f7931a','#00ffa3'],idx=0;
    for(var s in prices){lb.push(s);var bal=assets[s]?assets[s].virtual_balance:10000;vl.push(bal);idx++;}
    if(charts.alloc)charts.alloc.destroy();
    charts.alloc=new Chart(cv,{type:'doughnut',data:{labels:lb,datasets:[{data:vl,backgroundColor:bg,borderColor:'#0a1628',borderWidth:3,hoverOffset:8}]},
        options:{responsive:true,maintainAspectRatio:false,cutout:'65%',plugins:{legend:{position:'bottom',labels:{color:'#7b93b8',font:{family:'Inter',size:12},padding:16}},
        tooltip:{callbacks:{label:function(ctx){return ctx.label+': $'+fN(ctx.parsed,2);}}}}}});
}

/* ═══ HISTORY (CSV log) ═══ */
function fetchHistory(){
    var endpoint=currentVersion==='v209'?'/api/history?limit=500':'/api/v21/history?limit=500';
    fetch(endpoint).then(function(r){if(!r.ok)throw new Error();return r.json();})
    .then(function(d){historyData[currentVersion]=d;renderTradeLog();renderCharts();})
    .catch(function(e){console.warn('History:',e);});
}

function renderTradeLog(){
    var tb=document.querySelector('#trade-log-table tbody');if(!tb)return;
    var f=document.getElementById('log-asset-filter'),af=f?f.value:'';
    var fd=historyData[currentVersion]||[];if(af)fd=fd.filter(function(r){return r.asset===af;});
    var numC=['proba_high','position_size','price_close','pnl','virtual_balance','current_dd','realized_vol','vol_ratio','dd_scalar','latency_ms','edge_proba'];
    fd=fd.slice();
    fd.sort(function(a,b){
        var va=a[sortCol]||'',vb=b[sortCol]||'';
        if(numC.indexOf(sortCol)>=0){va=parseFloat(va)||0;vb=parseFloat(vb)||0;}
        if(va<vb)return sortDir==='asc'?-1:1;if(va>vb)return sortDir==='asc'?1:-1;return 0;
    });
    var h='';
    for(var i=0;i<Math.min(fd.length,200);i++){
        var row=fd[i];
        var pnlVal=parseFloat(row.pnl||0);
        var balVal=parseFloat(row.virtual_balance||0);
        var priceVal=parseFloat(row.price_close||0);
        var probaVal=parseFloat(row.proba_high||0);
        var posVal=parseFloat(row.position_size||0);
        var ddVal=parseFloat(row.current_dd||0);
        var rvol=parseFloat(row.realized_vol||0);
        var vratio=parseFloat(row.vol_ratio||0);
        var ddsc=parseFloat(row.dd_scalar||0);
        var latMs=parseFloat(row.latency_ms||0);
        var pnlPct=balVal>0?((balVal-10000)/10000*100):0;
        var pc=pnlVal>=0?'positive':'negative';
        var ts=row.timestamp?new Date(row.timestamp).toLocaleString('es-CU',{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit',timeZone:'America/Havana'}):'--';

        // V21 edge state
        var edgeHtml='';
        if(currentVersion==='v21'){
            var edgeState=row.edge_state||'N/A';
            var edgeProba=parseFloat(row.edge_proba||0);
            var edgeColor='var(--text-secondary)';
            if(edgeState==='EDGE_ON')edgeColor='var(--green)';
            if(edgeState==='EDGE_OFF')edgeColor='var(--red)';
            edgeHtml='<td style="color:'+edgeColor+'">'+edgeState+'</td><td>'+edgeProba.toFixed(4)+'</td>';
        }

        h+='<tr>'
          +'<td>'+ts+'</td>'
          +'<td><strong>'+( row.asset||'--')+'</strong></td>'
          +'<td>'+probaVal.toFixed(3)+'</td>'
          +'<td>'+(posVal*100).toFixed(1)+'%</td>'
          +'<td>$'+fN(priceVal,2)+'</td>'
          +'<td class="'+pc+'">'+pnlPct.toFixed(2)+'%</td>'
          +'<td class="'+pc+'">$'+fN(pnlVal,2)+'</td>'
          +'<td>$'+fN(balVal,2)+'</td>'
          +'<td>'+(ddVal*100).toFixed(2)+'%</td>'
          +'<td>'+rvol.toFixed(3)+'</td>'
          +'<td>'+vratio.toFixed(2)+'</td>'
          +'<td>'+ddsc.toFixed(2)+'</td>'
          +'<td>'+latMs.toFixed(0)+'ms</td>'
          +edgeHtml
          +'</tr>';
    }
    var colCount=currentVersion==='v21'?15:13;
    tb.innerHTML=h||'<tr><td colspan="'+colCount+'" style="text-align:center;color:#4a6080;padding:40px;">Esperando datos — primera barra 4H en progreso</td></tr>';
    var ce=document.getElementById('trade-count');if(ce)ce.textContent=fd.length+' registros';
}

function initSorting(){
    document.querySelectorAll('#trade-log-table th[data-sort]').forEach(function(th){
        th.addEventListener('click',function(){
            var c=th.dataset.sort;
            if(sortCol===c)sortDir=sortDir==='asc'?'desc':'asc';else{sortCol=c;sortDir='desc';}
            document.querySelectorAll('#trade-log-table th').forEach(function(h2){h2.classList.remove('sorted-asc','sorted-desc');});
            th.classList.add(sortDir==='asc'?'sorted-asc':'sorted-desc');renderTradeLog();
        });
    });
}

/* ═══ CHARTS ═══ */
function renderCharts(){
    var hd=historyData[currentVersion]||[];
    if(hd.length===0)return;
    mkChart('proba-chart','proba_high',false,0,1);
    mkChart('position-chart','position_size',true,0,0.6);
    mkChart('balance-chart','virtual_balance',false);
    mkChart('drawdown-chart','current_dd',true,undefined,0,true);
}

function mkChart(canvasId,field,fill,yMin,yMax,reverse){
    var cv=document.getElementById(canvasId);if(!cv)return;
    var colors={'ETH/USDT':'#627eea','BTC/USDT':'#f7931a','SOL/USDT':'#00ffa3'};
    var datasets=[];
    var hd=historyData[currentVersion]||[];
    for(var asset in colors){
        var ad=hd.filter(function(r){return r.asset===asset;});
        if(ad.length===0)continue;
        var multiplier=(field==='current_dd'||field==='position_size')?100:1;
        datasets.push({
            label:asset,
            data:ad.map(function(r){return parseFloat(r[field]||0)*multiplier;}),
            borderColor:colors[asset],
            backgroundColor:colors[asset]+(fill?'20':'00'),
            borderWidth:1.5,pointRadius:ad.length<20?3:0,
            fill:!!fill,tension:0.3
        });
    }
    var first=null;
    for(var a in colors){if(hd.some(function(r){return r.asset===a;})){first=a;break;}}
    var refData=first?hd.filter(function(r){return r.asset===first;}):hd;
    var labels=refData.map(function(r){
        if(!r.timestamp)return'';var d=new Date(r.timestamp);
        return d.toLocaleDateString('es-CU',{month:'short',day:'numeric',timeZone:'America/Havana'})+' '
              +d.toLocaleTimeString('es-CU',{hour:'2-digit',minute:'2-digit',timeZone:'America/Havana'});
    });
    var key=canvasId.replace('-chart','');if(charts[key])charts[key].destroy();
    var yOpts={ticks:{color:'#4a6080',font:{family:'Inter',size:10}},grid:{color:'rgba(13,40,71,0.5)'}};
    if(yMin!==undefined)yOpts.min=yMin;if(yMax!==undefined)yOpts.max=yMax;if(reverse)yOpts.reverse=true;
    var suffix='';
    if(field==='current_dd'||field==='position_size')suffix='%';
    if(field==='virtual_balance')suffix=' USD';
    charts[key]=new Chart(cv,{type:'line',data:{labels:labels,datasets:datasets},options:{
        responsive:true,maintainAspectRatio:false,animation:{duration:500},
        interaction:{mode:'index',intersect:false},
        scales:{x:{ticks:{color:'#4a6080',font:{family:'Inter',size:10},maxRotation:0,maxTicksLimit:10},grid:{color:'rgba(13,40,71,0.5)'}},y:yOpts},
        plugins:{legend:{labels:{color:'#7b93b8',font:{family:'Inter',size:11},boxWidth:12}},
            tooltip:{callbacks:{label:function(ctx){return ctx.dataset.label+': '+ctx.parsed.y.toFixed(field==='virtual_balance'?2:3)+suffix;}}}
        }}});
}

/* ═══ SYSTEM ═══ */
function fetchSystem(){
    fetch('/api/system').then(function(r){if(!r.ok)throw new Error();return r.json();})
    .then(function(d){renderSystem(d);})
    .catch(function(e){console.warn('System:',e);});
}

function renderSystem(d){
    setMeter('cpu',d.cpu_percent);
    if(d.memory){setMeter('memory',d.memory.percent);el('memory-detail',d.memory.used_gb.toFixed(1)+' GB / '+d.memory.total_gb.toFixed(1)+' GB');}
    if(d.disk){setMeter('disk',d.disk.percent);el('disk-detail',d.disk.used_gb.toFixed(1)+' GB / '+d.disk.total_gb.toFixed(1)+' GB');}

    // Update service badges
    var ss=d.service_status||{};
    var v209Badge=document.getElementById('service-v209');
    var v21Badge=document.getElementById('service-v21');
    if(v209Badge){
        var v209Active=ss['orion-v209']==='active';
        v209Badge.textContent='V20.9: '+(v209Active?'Activo':ss['orion-v209']||'--');
        v209Badge.className='badge '+(v209Active?'badge-success':'badge-warning');
    }
    if(v21Badge){
        var v21Active=ss['orion-v21']==='active';
        v21Badge.textContent='V21.0: '+(v21Active?'Activo':ss['orion-v21']||'--');
        v21Badge.className='badge '+(v21Active?'badge-success':'badge-warning');
    }

    var se=document.getElementById('service-status-detail');
    if(se){var ia=ss['orion-v209']==='active';se.textContent=ia?'Activo':ss['orion-v209']||'--';se.className='badge '+(ia?'badge-success':'badge-warning');}
    el('hostname',d.hostname||'--');el('uptime',d.uptime||'--');
}

function setMeter(n,p){
    var b=document.getElementById(n+'-meter'),v=document.getElementById(n+'-value');
    if(b){b.style.width=Math.max(p,1.5)+'%';
        if(p>90)b.style.background='linear-gradient(90deg,#ff4757,#ff6b81)';
        else if(p>70)b.style.background='linear-gradient(90deg,#ffa726,#ffb74d)';
        else b.style.background='linear-gradient(90deg,#0077ff,#00a8ff)';}
    if(v)v.textContent=p.toFixed(1)+'%';
}

/* ═══ UTILS ═══ */
function fN(n,d){if(isNaN(n))return'0.00';return n.toLocaleString('en-US',{minimumFractionDigits:d,maximumFractionDigits:d});}
function fC(n){if(n>=1e9)return(n/1e9).toFixed(1)+'B';if(n>=1e6)return(n/1e6).toFixed(1)+'M';if(n>=1e3)return(n/1e3).toFixed(1)+'K';return n.toFixed(0);}
function el(id,v){var e=document.getElementById(id);if(e)e.textContent=v;}
function anim(id,nv){var e=document.getElementById(id);if(!e)return;if(e.textContent!==nv){e.style.transition='opacity 0.15s';e.style.opacity='0.4';setTimeout(function(){e.textContent=nv;e.style.opacity='1';},150);}}
function updateBadge(s){var b=document.getElementById('service-status');if(!b)return;if(s==='activo'){b.textContent='Conectado';b.className='badge badge-success';}else{b.textContent='Desconectado';b.className='badge badge-error';}}
function setAutoReload(a){var e=document.getElementById('auto-reload');if(!e)return;e.innerHTML=a?'<span class="pulse-dot"></span> Auto-recarga activa':'<span class="pulse-dot inactive"></span> Pausado';}
})();