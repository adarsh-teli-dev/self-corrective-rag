
import { useEffect, useState } from "react";
import {
  Activity, ArrowUpRight, BookOpen, Check,
  CheckCircle2, ChevronDown, CircleHelp, Clock3, Database,
  ExternalLink, FileSearch, Globe2, History, Layers3, LoaderCircle,
  Menu, MessageSquareText, PanelLeftClose, Plus, Send, ShieldCheck,
  Sparkles, Waypoints, X, Zap
} from "lucide-react";

const API = "http://127.0.0.1:8000";

type Source = {
  title?: string;
  url?: string;
  source?: string;
  page?: number | null;
  chunk_id?: string | null;
  content?: string;
};

type ClaimCheck = {
  claim?: string;
  supported?: boolean;
  supporting_evidence_ids?: number[];
  reasoning?: string;
};

type Verification = {
  verified?: boolean;
  citation_check?: boolean;
  overall_status?: string;
  summary?: string;
  checks?: ClaimCheck[];
};

type PipelineResponse = {
  question: string;
  answer: string;
  route_taken?: string;
  evidence_count?: number;
  final_action?: string;
  verification?: Verification | null;
  sources?: Source[];
  initial_evaluation?: {
    decision?: string;
    relevance_score?: number;
    evidence_sufficient?: boolean;
    reasoning?: string;
  } | null;
  initial_route?: {
    action?: string;
    reason?: string;
  } | null;
  verification_history?: unknown[];
  correction_attempts?: number;
  elapsed_seconds?: number;
};

const examples = [
  "How does Corrective Retrieval Augmented Generation work?",
  "How does RAGAS evaluate retrieval-augmented generation systems?",
  "What happens when retrieved documents are irrelevant?",
];

function pretty(value?: string) {
  if (!value) return "Not reported";
  return value.replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase());
}

function App() {
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<PipelineResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [apiOnline, setApiOnline] = useState(false);
  const [activeTab, setActiveTab] = useState("Ask AI");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [showTrace, setShowTrace] = useState(true);
  const [expandedSource, setExpandedSource] = useState<number | null>(null);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const response = await fetch(`${API}/api/health`);
        if (mounted) setApiOnline(response.ok);
      } catch {
        if (mounted) setApiOnline(false);
      }
    };
    void check();
    const timer = window.setInterval(check, 15000);
    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  async function ask(value = question) {
    const q = value.trim();
    if (!q || loading) return;

    setQuestion(q);
    setLoading(true);
    setError("");
    setResult(null);
    setExpandedSource(null);
    setActiveTab("Ask AI");

    try {
      const response = await fetch(`${API}/api/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail || "The request failed.");
      }

      setResult(data as PipelineResponse);
    } catch (e) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not connect to the RAG backend."
      );
    } finally {
      setLoading(false);
    }
  }

  const verified = result?.verification?.verified === true;
  const status = result?.verification?.overall_status ??
    (verified ? "verified" : "not_verified");

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <aside className="sidebar">
          <div className="brand">
            <div className="brand-mark"><ShieldCheck size={23} /></div>
            <div>
              <div className="brand-name">CRAG<span></span></div>
              <div className="brand-subtitle">RETRIEVAL INTELLIGENCE</div>
            </div>
            <button className="icon-button sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Collapse sidebar">
              <PanelLeftClose size={18} />
            </button>
          </div>

          <div className="workspace-label">WORKSPACE</div>
          <nav className="nav-list">
            <button className={`nav-item ${activeTab === "Ask AI" ? "active" : ""}`} onClick={() => setActiveTab("Ask AI")}>
              <MessageSquareText size={18} /> <span>Ask AI</span><span className="nav-dot" />
            </button>
            <button className={`nav-item ${activeTab === "Evidence" ? "active" : ""}`} onClick={() => setActiveTab("Evidence")}>
              <FileSearch size={18} /> <span>Evidence explorer</span>
            </button>
            <button className={`nav-item ${activeTab === "Benchmarks" ? "active" : ""}`} onClick={() => setActiveTab("Benchmarks")}>
              <Activity size={18} /> <span>Benchmark lab</span>
            </button>
            <button className={`nav-item ${activeTab === "History" ? "active" : ""}`} onClick={() => setActiveTab("History")}>
              <History size={18} /> <span>Session history</span>
            </button>
          </nav>

          <div className="sidebar-divider" />
          <div className="workspace-label">SYSTEM COMPONENTS</div>
          <div className="component-list">
            <div className="component-row"><Database size={16} /><span>Chroma vector store</span><i className="status-dot" /></div>
            <div className="component-row"><Sparkles size={16} /><span>Qwen · Ollama</span><i className="status-dot" /></div>
            <div className="component-row"><Globe2 size={16} /><span>Tavily web search</span><i className="status-dot" /></div>
            <div className="component-row"><ShieldCheck size={16} /><span>Claim verification</span><i className="status-dot" /></div>
          </div>

          <div className="sidebar-bottom">
            <div className="connection-card">
              <div className={`connection-indicator ${apiOnline ? "online" : ""}`} />
              <div className="connection-copy">
                <strong>{apiOnline ? "Backend connected" : "Backend offline"}</strong>
                <span>{apiOnline ? "FastAPI · localhost:8000" : "Check your API terminal"}</span>
              </div>
            </div>
            <div className="user-note"><div className="avatar">AI</div><div><strong>Research workspace</strong><span>Local inference enabled</span></div></div>
          </div>
        </aside>
      )}

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            {!sidebarOpen && <button className="icon-button" onClick={() => setSidebarOpen(true)} aria-label="Open sidebar"><Menu size={19} /></button>}
            <span className="breadcrumb-muted">Workspace</span><span className="breadcrumb-separator">/</span><strong>{activeTab}</strong>
          </div>
          <div className="topbar-right">
            <span className="local-badge"><span /> LOCAL INFERENCE</span>
            <button className="icon-button help-button" title="Ask a question to begin"><CircleHelp size={18} /></button>
          </div>
        </header>

        {activeTab === "Ask AI" && (
          <div className="page-content">
            <section className="hero">
              <div className="eyebrow"><span className="eyebrow-line" /> SELF-CORRECTIVE RAG <span className="eyebrow-version">RESEARCH PREVIEW</span></div>
              <h1>Answers should<br /> <span>earn your trust.</span></h1>
              <p className="hero-description">Retrieve evidence. Evaluate relevance. Correct the path.<br className="desktop-break" /> Verify what the model tells you.</p>

              <div className="question-card">
                <div className="question-card-top">
                  <Sparkles size={17} /><span>ASK YOUR KNOWLEDGE ENGINE</span>
                  <span className="question-card-hint">Enter ↵ to submit</span>
                </div>
                <textarea
                  value={question}
                  onChange={e => setQuestion(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      void ask();
                    }
                  }}
                  placeholder="Ask a question. Your evidence trail starts here..."
                  rows={3}
                  disabled={loading}
                />
                <div className="question-card-bottom">
                  <div className="mode-chips"><span><Database size={13} /> Local retrieval</span><span><Globe2 size={13} /> Web fallback</span><span><ShieldCheck size={13} /> Verification</span></div>
                  <button className="ask-button" disabled={loading || !question.trim() || !apiOnline} onClick={() => void ask()}>
                    {loading ? <LoaderCircle className="spin" size={16} /> : <Send size={15} />}
                    {loading ? "Working..." : "Ask Verity"}
                  </button>
                </div>
              </div>

              {!result && !loading && (
                <div className="suggestions">
                  <span className="suggestion-label">TRY ASKING</span>
                  {examples.map((example, i) => (
                    <button key={example} className="suggestion" onClick={() => { setQuestion(example); void ask(example); }}>
                      <span className="suggestion-number">0{i + 1}</span>{example}<ArrowUpRight size={14} />
                    </button>
                  ))}
                </div>
              )}
            </section>

            {error && <div className="error-banner"><X size={18} /><div><strong>Couldn't complete the request</strong><p>{error}</p></div></div>}

            {loading && (
              <section className="loading-panel">
                <div className="loading-icon"><LoaderCircle size={24} className="spin" /></div>
                <div><h3>Following the evidence trail</h3><p>Retrieval, evaluation, routing and verification are running in your Python pipeline. This can take a little while.</p></div>
                <div className="loading-track"><span /></div>
              </section>
            )}

            {result && <Results result={result} verified={verified} status={status} showTrace={showTrace} setShowTrace={setShowTrace} expandedSource={expandedSource} setExpandedSource={setExpandedSource} />}

            <footer className="page-footer"><span><ShieldCheck size={14} /> Evidence-first generation</span><span>VERITY RESEARCH PLATFORM <i /> LOCAL MODEL</span></footer>
          </div>
        )}

        {activeTab === "Evidence" && (
          <div className="page-content secondary-page">
            <div className="eyebrow"><span className="eyebrow-line" /> KNOWLEDGE INSPECTION</div>
            <h1 className="secondary-title">Evidence <span>explorer.</span></h1>
            <p className="secondary-description">Inspect the source material from your most recent question.</p>
            {result ? <SourceList sources={result.sources ?? []} expandedSource={expandedSource} setExpandedSource={setExpandedSource} /> : <EmptyState icon={<BookOpen size={24} />} title="No evidence yet" description="Ask a question first to populate the evidence explorer." onClick={() => setActiveTab("Ask AI")} />}
          </div>
        )}

        {activeTab === "Benchmarks" && (
          <div className="page-content secondary-page">
            <div className="eyebrow"><span className="eyebrow-line" /> EVALUATION WORKSPACE</div>
            <h1 className="secondary-title">Benchmark <span>lab.</span></h1>
            <p className="secondary-description">Compare conventional RAG with your corrective pipeline using recorded evaluation runs.</p>
            <div className="notice-card"><Activity size={20} /><div><strong>Live benchmark integration is the next step</strong><p>These metrics are intentionally not fabricated. Your existing <code>evaluate.py</code> saves results under <code>data/evaluation/results/</code>. We'll add a backend endpoint to read those files and visualize actual results.</p></div></div>
            <div className="metric-grid">
              <Metric icon={<ShieldCheck size={19} />} label="Faithfulness" value="Awaiting results" />
              <Metric icon={<Clock3 size={19} />} label="Mean latency" value="Awaiting results" />
              <Metric icon={<Globe2 size={19} />} label="Fallback rate" value="Awaiting results" />
            </div>
          </div>
        )}

        {activeTab === "History" && (
          <div className="page-content secondary-page">
            <div className="eyebrow"><span className="eyebrow-line" /> RECENT ACTIVITY</div>
            <h1 className="secondary-title">Session <span>history.</span></h1>
            <p className="secondary-description">This view currently holds the latest result in this browser session.</p>
            {result ? <div className="history-card"><MessageSquareText size={19} /><div><strong>{result.question}</strong><p>{result.answer}</p><span>{result.elapsed_seconds ?? "?"} seconds · {pretty(result.route_taken)}</span></div><button className="text-button" onClick={() => { setQuestion(result.question); setActiveTab("Ask AI"); }}>Open <ArrowUpRight size={14} /></button></div> : <EmptyState icon={<History size={24} />} title="Your session is quiet" description="Questions you run will appear here during this session." onClick={() => setActiveTab("Ask AI")} />}
          </div>
        )}
      </main>
    </div>
  );
}

function Results({ result, verified, status, showTrace, setShowTrace, expandedSource, setExpandedSource }: {
  result: PipelineResponse;
  verified: boolean;
  status: string;
  showTrace: boolean;
  setShowTrace: (value: boolean) => void;
  expandedSource: number | null;
  setExpandedSource: (value: number | null) => void;
}) {
  const sources = result.sources ?? [];
  const checks = result.verification?.checks ?? [];

  return (
    <section className="results-section">
      <div className="result-heading">
        <div><div className="eyebrow"><span className="eyebrow-line" /> RESPONSE GENERATED</div><h2>Here's what the evidence says.</h2></div>
        <div className={`verification-pill ${verified ? "verified" : "unverified"}`}>{verified ? <CheckCircle2 size={15} /> : <ShieldCheck size={15} />}{pretty(status)}</div>
      </div>

      <div className="result-grid">
        <div className="answer-card">
          <div className="card-heading"><div className="card-heading-icon"><Sparkles size={17} /></div><div><strong>Generated answer</strong><span>Produced by your local RAG pipeline</span></div><span className="card-index">01 / ANSWER</span></div>
          <div className="answer-body">{result.answer || "The pipeline returned an empty answer."}</div>
          <div className="answer-meta">
            <span><Clock3 size={14} /> {result.elapsed_seconds ?? "?"}s elapsed</span>
            <span><Layers3 size={14} /> {result.evidence_count ?? 0} evidence items</span>
            <span><Zap size={14} /> {result.correction_attempts ?? 0} corrections</span>
          </div>
          <div className="card-divider" />
          <div className="card-subheading"><ShieldCheck size={16} /><strong>Claim verification</strong><span>{checks.length} checks</span></div>
          {checks.length ? checks.map((check, i) => (
            <div className="claim-row" key={`${check.claim}-${i}`}>
              <div className={`claim-status ${check.supported ? "supported" : "unsupported"}`}>{check.supported ? <Check size={13} /> : <X size={13} />}</div>
              <div className="claim-content"><p>{check.claim}</p><span>{check.supported ? "Marked as supported" : "Not supported by the verifier"}{check.supporting_evidence_ids?.length ? ` · Evidence ${check.supporting_evidence_ids.map(n => `[${n}]`).join(", ")}` : ""}</span>{check.reasoning && <details><summary>Verifier reasoning</summary><p>{check.reasoning}</p></details>}</div>
            </div>
          )) : <p className="muted-note">{result.verification?.summary || "The API did not return individual claim checks."}</p>}
          {result.verification?.summary && <div className="verification-summary"><ShieldCheck size={15} />{result.verification.summary}</div>}
        </div>

        <div className="right-stack">
          <div className="route-card">
            <div className="card-heading compact"><div className="card-heading-icon"><Waypoints size={17} /></div><div><strong>Retrieval journey</strong><span>Route selected by your system</span></div></div>
            <div className="route-timeline">
              <TimelineStep icon={<Database size={15} />} title="Local retrieval" detail={`${result.evidence_count ?? 0} final evidence items`} state="done" />
              <TimelineStep icon={<Activity size={15} />} title="Retrieval evaluator" detail={pretty(result.initial_evaluation?.decision)} state="done" />
              <TimelineStep icon={<Globe2 size={15} />} title={result.route_taken === "web_search" ? "Web fallback" : "Routing decision"} detail={pretty(result.route_taken)} state={result.route_taken === "web_search" ? "highlight" : "done"} />
              <TimelineStep icon={<Sparkles size={15} />} title="Answer generation" detail={pretty(result.final_action)} state="done" />
              <TimelineStep icon={<ShieldCheck size={15} />} title="Answer verification" detail={pretty(status)} state={verified ? "highlight" : "pending"} last />
            </div>
            <button className="trace-toggle" onClick={() => setShowTrace(!showTrace)}>{showTrace ? "Hide evaluation details" : "Show evaluation details"}<ChevronDown size={15} className={showTrace ? "rotate" : ""} /></button>
            {showTrace && <div className="evaluation-detail">
              <div><span>Evaluator decision</span><strong>{pretty(result.initial_evaluation?.decision)}</strong></div>
              <div><span>Evidence sufficient</span><strong>{result.initial_evaluation?.evidence_sufficient === undefined ? "Not reported" : result.initial_evaluation.evidence_sufficient ? "Yes" : "No"}</strong></div>
              <div><span>Initial route</span><strong>{pretty(result.initial_route?.action)}</strong></div>
              <div><span>Final action</span><strong>{pretty(result.final_action)}</strong></div>
              {result.initial_route?.reason && <p>{result.initial_route.reason}</p>}
            </div>}
          </div>

          <div className="sources-card">
            <div className="card-heading compact"><div className="card-heading-icon"><BookOpen size={17} /></div><div><strong>Sources & evidence</strong><span>{sources.length} sources returned by API</span></div></div>
            <SourceList sources={sources} expandedSource={expandedSource} setExpandedSource={setExpandedSource} />
          </div>
        </div>
      </div>
    </section>
  );
}

function TimelineStep({ icon, title, detail, state, last = false }: { icon: React.ReactNode; title: string; detail: string; state: "done" | "highlight" | "pending"; last?: boolean }) {
  return <div className={`timeline-step ${state} ${last ? "last" : ""}`}><div className="timeline-icon">{icon}</div><div className="timeline-copy"><strong>{title}</strong><span>{detail}</span></div>{state === "done" ? <CheckCircle2 className="timeline-check" size={16} /> : state === "highlight" ? <span className="live-mark" /> : <span className="pending-mark" />}</div>;
}

function SourceList({ sources, expandedSource, setExpandedSource }: { sources: Source[]; expandedSource: number | null; setExpandedSource: (value: number | null) => void }) {
  if (!sources.length) return <div className="empty-sources"><FileSearch size={22} /><strong>No source records returned</strong><p>The pipeline did not provide a source list for this request.</p></div>;

  return <div className="source-list">{sources.map((source, i) => {
    const expanded = expandedSource === i;
    return <div className="source-item" key={`${source.url ?? source.title ?? "source"}-${i}`}>
      <button className="source-main" onClick={() => setExpandedSource(expanded ? null : i)}>
        <div className={`source-number ${source.source === "web" ? "web" : ""}`}>{String(i + 1).padStart(2, "0")}</div>
        <div className="source-info"><strong>{source.title || source.chunk_id || "Untitled source"}</strong><span>{source.source === "web" ? "WEB SOURCE" : source.source || "SOURCE"}{source.page ? ` · PAGE ${source.page}` : ""}</span></div>
        <ChevronDown size={15} className={expanded ? "rotate" : ""} />
      </button>
      {expanded && <div className="source-expanded">
        {source.url && <a className="source-url" href={source.url} target="_blank" rel="noreferrer">{source.url}<ExternalLink size={13} /></a>}
        {source.chunk_id && <p>Chunk: {source.chunk_id}</p>}
        {source.content && <p className="source-excerpt">{source.content}</p>}
        {!source.url && !source.content && <p className="muted-note">No URL or passage content was returned for this source.</p>}
      </div>}
    </div>;
  })}</div>;
}

function Metric({ icon, label, value }: { icon: React.ReactNode; label: string; value: string }) {
  return <div className="metric-card"><div className="metric-icon">{icon}</div><span>{label}</span><strong>{value}</strong></div>;
}

function EmptyState({ icon, title, description, onClick }: { icon: React.ReactNode; title: string; description: string; onClick: () => void }) {
  return <div className="empty-state"><div className="empty-state-icon">{icon}</div><h3>{title}</h3><p>{description}</p><button className="secondary-button" onClick={onClick}><Plus size={15} /> Start a question</button></div>;
}

export default App;
