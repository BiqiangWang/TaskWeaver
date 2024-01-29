"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[9285],{3734:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>r,contentTitle:()=>a,default:()=>h,frontMatter:()=>l,metadata:()=>o,toc:()=>d});var s=t(5893),i=t(1151);const l={},a="GLM",o={id:"llms/glm",title:"GLM",description:"1. GLM (ChatGLM) is a LLM developed by Zhipu AI and Tsinghua KEG. Go to ZhipuAI and register an account and get the API key. More details can be found here.",source:"@site/docs/llms/glm.md",sourceDirName:"llms",slug:"/llms/glm",permalink:"/TaskWeaver/docs/llms/glm",draft:!1,unlisted:!1,editUrl:"https://github.com/microsoft/TaskWeaver/tree/docs/website/docs/llms/glm.md",tags:[],version:"current",frontMatter:{},sidebar:"documentSidebar",previous:{title:"QWen",permalink:"/TaskWeaver/docs/llms/qwen"},next:{title:"Plugin Introduction",permalink:"/TaskWeaver/docs/plugin/plugin_intro"}},r={},d=[];function c(e){const n={a:"a",code:"code",h1:"h1",li:"li",ol:"ol",p:"p",pre:"pre",...(0,i.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h1,{id:"glm",children:"GLM"}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["GLM (ChatGLM) is a LLM developed by Zhipu AI and Tsinghua KEG. Go to ",(0,s.jsx)(n.a,{href:"https://open.bigmodel.cn/",children:"ZhipuAI"})," and register an account and get the API key. More details can be found ",(0,s.jsx)(n.a,{href:"https://open.bigmodel.cn/overview",children:"here"}),"."]}),"\n",(0,s.jsx)(n.li,{children:"Install the required packages dashscope."}),"\n"]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-bash",children:"pip install zhipuai\n"})}),"\n",(0,s.jsxs)(n.ol,{start:"3",children:["\n",(0,s.jsxs)(n.li,{children:["Add the following configuration to ",(0,s.jsx)(n.code,{children:"taskweaver_config.json"}),":"]}),"\n"]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-json",metastring:"showLineNumbers",children:'{\n  "llm.api_type": "zhipuai",\n  "llm.model": "glm-4",\n  "llm.embedding_model": "embedding-2",\n  "llm.embedding_api_type": "zhipuai",\n  "llm.api_key": "YOUR_API_KEY"\n}\n'})}),"\n",(0,s.jsxs)(n.p,{children:["NOTE: ",(0,s.jsx)(n.code,{children:"llm.model"})," is the model name of zhipuai  API.\nYou can find the model name in the ",(0,s.jsx)(n.a,{href:"https://open.bigmodel.cn/dev/api#language",children:"GLM model list"}),"."]}),"\n",(0,s.jsxs)(n.ol,{start:"4",children:["\n",(0,s.jsxs)(n.li,{children:["Start TaskWeaver and chat with TaskWeaver.\nYou can refer to the ",(0,s.jsx)(n.a,{href:"/TaskWeaver/docs/quickstart",children:"Quick Start"})," for more details."]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(c,{...e})}):c(e)}},1151:(e,n,t)=>{t.d(n,{Z:()=>o,a:()=>a});var s=t(7294);const i={},l=s.createContext(i);function a(e){const n=s.useContext(l);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function o(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:a(e.components),s.createElement(l.Provider,{value:n},e.children)}}}]);