"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[6366],{2417:(e,n,o)=>{o.r(n),o.d(n,{assets:()=>c,contentTitle:()=>a,default:()=>h,frontMatter:()=>s,metadata:()=>i,toc:()=>l});var r=o(4848),t=o(8453);const s={},a="GroqChat",i={id:"llms/groq",title:"GroqChat",description:"1. Groq was founded in 2016 by Chief Executive Officer Jonathan Ross, a former Google LLC engineer who invented the search giant's TPU machine learning processors. Go to Groq and register an account and get the API key from here. More details can be found here.",source:"@site/docs/llms/groq.md",sourceDirName:"llms",slug:"/llms/groq",permalink:"/TaskWeaver/docs/llms/groq",draft:!1,unlisted:!1,editUrl:"https://github.com/microsoft/TaskWeaver/tree/main/website/docs/llms/groq.md",tags:[],version:"current",frontMatter:{}},c={},l=[];function d(e){const n={a:"a",admonition:"admonition",code:"code",h1:"h1",header:"header",li:"li",ol:"ol",p:"p",pre:"pre",...(0,t.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"groqchat",children:"GroqChat"})}),"\n",(0,r.jsxs)(n.ol,{children:["\n",(0,r.jsxs)(n.li,{children:["Groq was founded in 2016 by Chief Executive Officer ",(0,r.jsx)(n.code,{children:"Jonathan Ross"}),", a former Google LLC engineer who invented the search giant's TPU machine learning processors. Go to ",(0,r.jsx)(n.a,{href:"https://groq.com/",children:"Groq"})," and register an account and get the API key from ",(0,r.jsx)(n.a,{href:"https://console.groq.com/keys",children:"here"}),". More details can be found ",(0,r.jsx)(n.a,{href:"https://console.groq.com/docs/quickstart",children:"here"}),"."]}),"\n",(0,r.jsxs)(n.li,{children:["Install the required packages ",(0,r.jsx)(n.code,{children:"groq"}),"."]}),"\n"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-bash",children:"pip install groq\n"})}),"\n",(0,r.jsxs)(n.ol,{start:"3",children:["\n",(0,r.jsxs)(n.li,{children:["Add the following configuration to ",(0,r.jsx)(n.code,{children:"taskweaver_config.json"}),":"]}),"\n"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-json",metastring:"showLineNumbers",children:'{\n    "llm.api_base": "https://console.groq.com/",\n    "llm.api_key": "YOUR_API_KEY",\n    "llm.api_type": "groq",\n    "llm.model": "mixtral-8x7b-32768"\n}\n'})}),"\n",(0,r.jsx)(n.admonition,{type:"tip",children:(0,r.jsxs)(n.p,{children:["NOTE: ",(0,r.jsx)(n.code,{children:"llm.model"})," is the model name of Groq LLM API.\nYou can find the model name in the ",(0,r.jsx)(n.a,{href:"https://console.groq.com/docs/models",children:"Groq LLM model list"}),"."]})}),"\n",(0,r.jsxs)(n.ol,{start:"4",children:["\n",(0,r.jsxs)(n.li,{children:["Start TaskWeaver and chat with TaskWeaver.\nYou can refer to the ",(0,r.jsx)(n.a,{href:"/TaskWeaver/docs/quickstart",children:"Quick Start"})," for more details."]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,t.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(d,{...e})}):d(e)}},8453:(e,n,o)=>{o.d(n,{R:()=>a,x:()=>i});var r=o(6540);const t={},s=r.createContext(t);function a(e){const n=r.useContext(s);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function i(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:a(e.components),r.createElement(s.Provider,{value:n},e.children)}}}]);