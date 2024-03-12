"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[8490],{8715:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>c,contentTitle:()=>o,default:()=>d,frontMatter:()=>s,metadata:()=>i,toc:()=>h});var r=n(5893),a=n(1151);const s={},o="Tracing",i={id:"tracing",title:"Tracing",description:"TaskWeaver now supports tracing with OpenTelemetry,",source:"@site/docs/tracing.md",sourceDirName:".",slug:"/tracing",permalink:"/TaskWeaver/docs/tracing",draft:!1,unlisted:!1,editUrl:"https://github.com/microsoft/TaskWeaver/tree/main/website/docs/tracing.md",tags:[],version:"current",frontMatter:{},sidebar:"documentSidebar",previous:{title:"CLI Only Mode",permalink:"/TaskWeaver/docs/cli_only"}},c={},h=[{value:"How to enable tracing",id:"how-to-enable-tracing",level:2},{value:"How to customize tracing",id:"how-to-customize-tracing",level:2}];function l(e){const t={a:"a",code:"code",h1:"h1",h2:"h2",img:"img",li:"li",p:"p",pre:"pre",ul:"ul",...(0,a.a)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(t.h1,{id:"tracing",children:"Tracing"}),"\n",(0,r.jsx)(t.p,{children:"TaskWeaver now supports tracing with OpenTelemetry,\nwhich is one of the most popular open-source observability frameworks. This allows you to trace the following:"}),"\n",(0,r.jsxs)(t.ul,{children:["\n",(0,r.jsx)(t.li,{children:"Interactions between roles, i.e., the Planner, the CodeInterpreter, and the Executor."}),"\n",(0,r.jsx)(t.li,{children:"The time consumed by each role and major components of TaskWeaver."}),"\n",(0,r.jsx)(t.li,{children:"The prompts sent to the LLM and the responses received from the LLM."}),"\n",(0,r.jsx)(t.li,{children:"The status of the tasks and the errors encountered."}),"\n"]}),"\n",(0,r.jsx)(t.p,{children:"The following screenshot shows a trace of a simple task: analyzing an uploaded file."}),"\n",(0,r.jsx)(t.p,{children:(0,r.jsx)(t.img,{alt:"Tracing",src:n(6645).Z+"",width:"1874",height:"662"})}),"\n",(0,r.jsx)(t.p,{children:"From this view, you can see the timeline of the task execution, which breaks majorly into\nthree parts:"}),"\n",(0,r.jsxs)(t.ul,{children:["\n",(0,r.jsx)(t.li,{children:"The planning phase, where the Planner decides the sub-tasks to be executed."}),"\n",(0,r.jsx)(t.li,{children:"The code generation and execution phase, where the CodeGenerator generates the code and the CodeExecutor executes it."}),"\n",(0,r.jsx)(t.li,{children:"The reply phase, where the Planner sends the reply to the user."}),"\n"]}),"\n",(0,r.jsx)(t.p,{children:"The bars with a black line represent the critical path of the task execution, which is the longest path through the task execution.\nThis is useful for identifying the bottleneck of the task execution.\nWe can clearly see that, currently, the task execution is dominated by the calls to the LLM."}),"\n",(0,r.jsx)(t.p,{children:"We can click the span (a unit of work in the trace) to see the details of the span, including the logs and the attributes."}),"\n",(0,r.jsx)(t.p,{children:"The screenshot below shows the prompt of the CodeGenerator to the LLM:"}),"\n",(0,r.jsx)(t.p,{children:(0,r.jsx)(t.img,{alt:"Tracing Prompt",src:n(4525).Z+"",width:"1619",height:"658"})}),"\n",(0,r.jsx)(t.p,{children:"We also recorded the generated code, the posts between different roles, etc. in the trace."}),"\n",(0,r.jsx)(t.p,{children:"There are also views of the trace, for example the call graph view, which shows the call hierarchy of the spans.\nHere is the call graph of the trace:"}),"\n",(0,r.jsx)(t.p,{children:(0,r.jsx)(t.img,{alt:"Tracing Call Graph",src:n(5900).Z+"",width:"1739",height:"303"})}),"\n",(0,r.jsx)(t.h2,{id:"how-to-enable-tracing",children:"How to enable tracing"}),"\n",(0,r.jsxs)(t.p,{children:["Tracing is by default disabled. To enable tracing, you need to install packages required by OpenTelemetry.\nPlease check the ",(0,r.jsx)(t.a,{href:"https://opentelemetry.io/docs/languages/python/",children:"OpenTelemetry website"})," for the installation guide.\nIt basically requires you to install the ",(0,r.jsx)(t.code,{children:"opentelemetry-api"}),", ",(0,r.jsx)(t.code,{children:"opentelemetry-sdk"}),", ",(0,r.jsx)(t.code,{children:"opentelemetry-exporter-otlp"}),",\nand ",(0,r.jsx)(t.code,{children:"opentelemetry-instrumentation"})," packages.\nAfter installing the packages, you can enable tracing by setting the ",(0,r.jsx)(t.code,{children:"tracing.enabled=true"})," in the project configuration file."]}),"\n",(0,r.jsxs)(t.p,{children:["Next, you need to set up a trace collector and a frontend to collect and view the traces. We recommend using ",(0,r.jsx)(t.a,{href:"https://www.jaegertracing.io/",children:"Jaeger"}),",\nwhich is a popular open-source tracing system.\nTo start, please visit the ",(0,r.jsx)(t.a,{href:"https://www.jaegertracing.io/docs/getting-started/",children:"Getting Started"}),' page of Jaeger.\nAn "All-in-one" Docker image is available, which is easy to start and use.\nThis docker image includes both the OpenTelemetry collector and the Jaeger frontend.\nIf the container is running at the same host as the TaskWeaver, you don\'t need to configure anything else.\nOtherwise, you need to set the ',(0,r.jsx)(t.code,{children:"tracing.endpoint"})," in the project configuration file to the endpoint of the OpenTelemetry collector.\nThe default endpoint of the OpenTelemetry collector is ",(0,r.jsx)(t.code,{children:"http://127.0.0.1:4318/v1/traces"}),"."]}),"\n",(0,r.jsxs)(t.p,{children:["After running the docker image, you can access the Jaeger frontend at ",(0,r.jsx)(t.code,{children:"http://localhost:16686"}),".\nNow, when you run TaskWeaver, issue a task, and access the Jaeger frontend, you can see the traces of the task execution.\nOn the left side panel, you can select the Service dropdown to filter the traces by the service name.\nThe service name of TaskWeaver is ",(0,r.jsx)(t.code,{children:"taskweaver.opentelemetry.tracer"}),"."]}),"\n",(0,r.jsx)(t.h2,{id:"how-to-customize-tracing",children:"How to customize tracing"}),"\n",(0,r.jsxs)(t.p,{children:["The instrumentation of TaskWeaver is done by the OpenTelemetry Python SDK.\nSo, if you want to customize the tracing, you need to modify the TaskWeaver code.\nIn TaskWeaver, we add a layer of abstraction to the OpenTelemetry SDK,\nso that it is easier to hide the details of the OpenTelemetry SDK from the TaskWeaver code.\nYou can find the abstraction layer in the ",(0,r.jsx)(t.code,{children:"taskweaver.module.tracing"})," module."]}),"\n",(0,r.jsxs)(t.p,{children:["In the ",(0,r.jsx)(t.code,{children:"taskweaver.module.tracing"})," module, we define the ",(0,r.jsx)(t.code,{children:"Tracing"})," class,\nwhich is a wrapper of the OpenTelemetry SDK. The ",(0,r.jsx)(t.code,{children:"Tracing"})," class provides the following methods:"]}),"\n",(0,r.jsxs)(t.ul,{children:["\n",(0,r.jsx)(t.li,{children:"set_span_status: Set the status of the span."}),"\n",(0,r.jsx)(t.li,{children:"set_span_attribute: Set the attribute of the span."}),"\n",(0,r.jsx)(t.li,{children:"set_span_exception: Set the exception of the span."}),"\n"]}),"\n",(0,r.jsxs)(t.p,{children:["In addition, we define the decorator ",(0,r.jsx)(t.code,{children:"tracing_decorator"})," (or the non-class version ",(0,r.jsx)(t.code,{children:"tracing_decorator_non_class"}),")\nto trace the function calls.\nWhen you need to create a context for tracing, you can use"]}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",children:'with get_tracer().start_as_current_span("span_name") as span:\n    # your code\n'})}),"\n",(0,r.jsx)(t.p,{children:"When you need to trace a function, you can use"}),"\n",(0,r.jsx)(t.pre,{children:(0,r.jsx)(t.code,{className:"language-python",children:"@tracing_decorator\ndef your_function(self, *args, **kwargs):\n    # your code\n"})})]})}function d(e={}){const{wrapper:t}={...(0,a.a)(),...e.components};return t?(0,r.jsx)(t,{...e,children:(0,r.jsx)(l,{...e})}):l(e)}},6645:(e,t,n)=>{n.d(t,{Z:()=>r});const r=n.p+"assets/images/trace-00cf7b585651ce69f5c2724248434eaa.png"},5900:(e,t,n)=>{n.d(t,{Z:()=>r});const r=n.p+"assets/images/trace_graph-9f532f2d7722c0510a3ce7748996a8ac.png"},4525:(e,t,n)=>{n.d(t,{Z:()=>r});const r=n.p+"assets/images/trace_prompt-bee25bb57232512704bbd9a21fa94674.png"},1151:(e,t,n)=>{n.d(t,{Z:()=>i,a:()=>o});var r=n(7294);const a={},s=r.createContext(a);function o(e){const t=r.useContext(s);return r.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function i(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:o(e.components),r.createElement(s.Provider,{value:t},e.children)}}}]);