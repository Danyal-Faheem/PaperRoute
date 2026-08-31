/*
 * htmx-lite — a tiny, dependency-free subset for PaperRoute polling.
 * Compatible with hx-get, hx-trigger="load, every Ns", hx-target and hx-swap="outerHTML".
 * This file is an original implementation for this project; no third-party runtime is loaded.
 */
(function(){'use strict';
  function parseEvery(value){var match=(value||'').match(/every\s+(\d+(?:\.\d+)?)\s*(ms|s)?/i);if(!match)return null;var n=parseFloat(match[1]);return match[2]&&match[2].toLowerCase()==='ms'?n:n*1000;}
  function targetFor(el){var selector=el.getAttribute('hx-target');return selector?document.querySelector(selector):el;}
  function request(el){var url=el.getAttribute('hx-get');if(!url)return;var target=targetFor(el);if(!target)return;fetch(url,{headers:{'HX-Request':'true','Accept':'text/html'},credentials:'same-origin'}).then(function(r){if(!r.ok)throw new Error('Request failed: '+r.status);return r.text();}).then(function(html){var swap=el.getAttribute('hx-swap')||'innerHTML';if(swap==='outerHTML'){var temp=document.createElement('template');temp.innerHTML=html.trim();var next=temp.content.firstElementChild;if(next){target.replaceWith(next);wire(next);}}else{target.innerHTML=html;wire(target);}}).catch(function(){target.classList.add('is-poll-error');});}
  function wire(root){var nodes=[];if(root&&root.nodeType===1)nodes.push(root);if(root&&root.querySelectorAll)nodes=nodes.concat(Array.prototype.slice.call(root.querySelectorAll('[hx-get]')));nodes.forEach(function(el){if(el.__paperrouteWired)return;el.__paperrouteWired=true;var trigger=el.getAttribute('hx-trigger')||'';if(/(?:^|,)\s*load(?:\s|,|$)/.test(trigger))request(el);var delay=parseEvery(trigger);if(delay){el.__paperrouteTimer=setInterval(function(){if(document.body.contains(el))request(el);else clearInterval(el.__paperrouteTimer);},delay);}});}
  document.addEventListener('DOMContentLoaded',function(){wire(document.body);});
})();
