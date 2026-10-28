import { useEffect, useMemo, useRef, useState } from 'react'
import './App.css'

async function getJSON(url, options) {
  const res = await fetch(url, options)
  if (!res.ok) {
    let detail = res.statusText
    try {
      detail = (await res.json()).error ?? detail
    } catch {
      /* keep statusText */
    }
    throw new Error(detail)
  }
  return res.json()
}

const post = (url, body) =>
  getJSON(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

const ms = (v) => (typeof v === 'number' ? v.toFixed(4) : '—')
const big = (v, d = 2) => (typeof v === 'number' ? v.toFixed(d) : '—')
const signed = (v) => `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`
const VERDICT_CLASS = { improved: 'good', degraded: 'bad', neutral: 'warn' }

/* The nine steps every run walks through, in plain language. */
const STAGES = [
  {
    id: 'input',
    title: 'Read the program',
    explain:
      'The compiler opens the .dsl file — plain text that describes the maths (here: attention followed by a two-layer MLP). Nothing is computed yet; we are reading characters and preparing the input tensors the GPU will work on.',
  },
  {
    id: 'tokens',
    title: 'Lexing — text into tokens',
    explain:
      'The text is split into "words" called tokens: keywords like const, names like Q, numbers like 64, punctuation like ->. Every compiler for every language starts here. Each token remembers its line and column, which is how errors can point at the exact spot.',
  },
  {
    id: 'parse',
    title: 'Parsing — tokens into a tree',
    explain:
      'A grammar (Parser.cup) says which sentences are legal in this little language. The parser checks the token stream against those rules and builds the tree shown here. This is the step the viva asks about: statements are pipelines ("out = …"), expressions are the pieces inside brackets and parentheses.',
  },
  {
    id: 'semantics',
    title: 'Understanding the maths',
    explain:
      'Shortcuts like attention(...) are expanded into basic operations (matrix multiply, softmax, …). Every shape is checked so the matrices line up, and the program becomes a graph of operations. This workload becomes 15 nodes — a handful of real operations plus the tensors supplied from outside.',
  },
  {
    id: 'fusion',
    title: 'The optimisation decision',
    explain:
      'Several operations can often be done in a single pass if the intermediate result never has to be written out to memory. The compiler prices every candidate ("merging these saves 32 KB of traffic and two launches") and either accepts or rejects it with a written reason. This is the core idea of the project — and nothing here is guessed, it is all logged.',
  },
  {
    id: 'emit',
    title: 'Writing GPU code',
    explain:
      'The compiler writes out CUDA C — the language the graphics card understands. Same idea as a normal compiler emitting machine code, except this output is readable text you can inspect. Nobody hand-wrote these kernels; they are generated.',
  },
  {
    id: 'build',
    title: 'Building for the card',
    explain:
      "NVIDIA's compiler (nvcc) turns that code into a program for this specific GPU. It also reports how much on-chip memory each kernel needs (registers / shared memory) and whether anything had to be 'spilled' to slow memory.",
  },
  {
    id: 'execute',
    title: 'Running on the GPU',
    explain:
      'The program runs many times over and a stopwatch (CUDA events) times every piece. We run two builds of the same program — "naive" (every operation separate) and "fused" (the compiler\'s plan) — so the numbers can be compared directly.',
  },
  {
    id: 'validate',
    title: 'Checking the answer',
    explain:
      'The same maths is recomputed in Python at double precision, and both GPU results are compared against it number by number. If they disagreed, this step would say FAIL — "it ran fast" is never accepted without "and it is still correct".',
  },
]

/* ---- token table (lexer output, shown in the pipeline) ---- */

const TOKEN_LITERALS = {
  SEMI: ';',
  COMMA: ',',
  COLON: ':',
  EQ: '=',
  ARROW: '->',
  LPAREN: '(',
  RPAREN: ')',
  LBRACKET: '[',
  RBRACKET: ']',
  CONST: 'const',
  DTYPE: 'dtype',
  STORE: 'store',
  ACCUM: 'accum',
  INPUT: 'input',
  PARAM: 'param',
  FP16: 'fp16',
  FP32: 'fp32',
}

function parseTokenLines(text) {
  const rows = []
  for (const line of text.split('\n')) {
    const m = line.match(/^\s*(\d+)\s+line\s+(\d+)\s+col\s+(\d+)\s+(\S+)\s*(.*)$/)
    if (m) rows.push({ n: m[1], line: m[2], col: m[3], type: m[4], value: m[5].trim() })
  }
  return rows
}

function TokenTable({ text }) {
  const rows = parseTokenLines(text)
  if (rows.length === 0) return <pre className="code small">{text}</pre>
  return (
    <div className="token-wrap">
      <table className="tokens">
        <thead>
          <tr>
            <th className="num">#</th>
            <th className="num">line:col</th>
            <th>token</th>
            <th>spelling</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.n}>
              <td className="num">{r.n}</td>
              <td className="num">
                {r.line}:{r.col}
              </td>
              <td className="mono">{r.type}</td>
              <td className="mono">{r.value || TOKEN_LITERALS[r.type] || ''}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

/* ------------------------------------------------------------------ dashboard */

function WorkloadCard({ w, selected, onSelect, disabled }) {
  return (
    <button
      type="button"
      className={`workload${selected ? ' selected' : ''}`}
      onClick={() => onSelect(w.id)}
      disabled={disabled}
    >
      <span className="workload-name">{w.id}</span>
      <span className="workload-dsl">{w.dsl}</span>
      <span className="workload-question">{w.question}</span>
      <span className="chips">
        {Object.entries(w.consts).map(([k, v]) => (
          <span className="chip" key={k}>
            {k}={v}
          </span>
        ))}
        <span className="chip">{w.tensors.length} tensors</span>
      </span>
    </button>
  )
}

function TensorTable({ tensors }) {
  return (
    <table>
      <thead>
        <tr>
          <th>tensor</th>
          <th>role</th>
          <th>stored as</th>
          <th className="num">shape</th>
        </tr>
      </thead>
      <tbody>
        {tensors.map((t) => (
          <tr key={t.name}>
            <td className="mono">{t.name}</td>
            <td>{t.kind === 'input' ? 'program input' : 'program parameter (weights)'}</td>
            <td className="mono">{t.dtype}</td>
            <td className="num">{t.shape.join(' × ')}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function DecisionList({ decisions }) {
  return (
    <ul className="decisions">
      {decisions.map((d, i) => (
        <li key={i}>
          <span className={`tag ${d.accepted ? 'good' : 'bad'}`}>
            {d.accepted ? 'FUSED' : 'REJECT'}
          </span>
          <span className="dec-kind">{d.kind}</span>
          <span className="dec-members">{d.members.join(', ')}</span>
          <p className="muted">{d.reason}</p>
        </li>
      ))}
    </ul>
  )
}

function RegionBars({ regions }) {
  const max = Math.max(...regions.flatMap((r) => [r.fused_ms, r.unfused_ms]), 1e-6)
  return (
    <div className="bars">
      {regions.map((r) => (
        <div className="bar-row" key={r.region}>
          <div className="bar-label">{r.region}</div>
          <div className="bar-track">
            <div className="bar-line">
              <div className="bar fused" style={{ width: `${(r.fused_ms / max) * 100}%` }} />
              <span className="bar-num">{ms(r.fused_ms)} ms fused</span>
            </div>
            <div className="bar-line">
              <div className="bar unfused" style={{ width: `${(r.unfused_ms / max) * 100}%` }} />
              <span className="bar-num">{ms(r.unfused_ms)} ms naive</span>
            </div>
          </div>
          <div className={`change ${VERDICT_CLASS[r.verdict]}`}>
            {signed(r.change_pct)}
            <small>{r.verdict}</small>
          </div>
        </div>
      ))}
    </div>
  )
}

function StageBody({ id, data, workload, onOpenFile }) {
  if (!data) return <p className="muted">waiting …</p>
  switch (id) {
    case 'input':
      return (
        <>
          <pre className="code">{data.source}</pre>
          <div className="chips">
            {Object.entries(data.consts ?? {}).map(([k, v]) => (
              <span className="chip" key={k}>
                {k}={v}
              </span>
            ))}
            <span className="chip">inputs generated with seed {data.seed}</span>
          </div>
          <TensorTable tensors={data.tensors ?? []} />
        </>
      )
    case 'tokens':
      return (
        <>
          <div className="chips">
            <span className="chip">{data.count} tokens</span>
            <span className="chip">each token knows its line and column</span>
          </div>
          <TokenTable text={data.text} />
        </>
      )
    case 'parse':
      return (
        <p className="muted">
          the CUP parser built the tree; its printout is produced by the Java front end — run
          <code> .\scripts\parse-tree.ps1 </code> in a terminal to see it.
        </p>
      )
    case 'semantics':
      return (
        <>
          <div className="chips">
            <span className="chip">{data.nodes} graph nodes</span>
            <span className="chip">{data.ops} operations</span>
            {(data.op_kinds ?? []).map((op) => (
              <span className="chip" key={op}>
                {op}
              </span>
            ))}
          </div>
          <p className="muted">
            annotated graph — placement says where each value lives: REGISTER / SHARED means it
            never goes out to main memory
          </p>
          <pre className="code small">{data.ir}</pre>
        </>
      )
    case 'fusion':
      return (
        <>
          <div className="chips">
            <span className="chip">
              {data.kernels_before} kernels → {data.kernels_after} kernels
            </span>
            <span className="chip">{data.accepted} accepted</span>
            <span className="chip">{data.rejected} rejected (with reasons)</span>
          </div>
          <DecisionList decisions={data.decision_log?.decisions ?? []} />
        </>
      )
    case 'emit':
      return (
        <div className="emit-grid">
          {['unfused', 'fused'].map((tag) =>
            data[tag] ? (
              <div className="emit-card" key={tag}>
                <div className="emit-title">{tag === 'fused' ? 'fused build' : 'naive build'}</div>
                <div className="mono">{data[tag].cu}</div>
                <div className="muted">
                  {data[tag].kernels} kernels · {Math.round(data[tag].bytes / 1024)} KB of
                  generated CUDA C
                </div>
                <button
                  onClick={() => onOpenFile(`generated CUDA C — ${tag}`, `/api/artifact/${workload}/${data[tag].cu}`)}
                >
                  view generated code
                </button>
              </div>
            ) : (
              <div className="emit-card" key={tag}>
                <div className="emit-title">{tag === 'fused' ? 'fused build' : 'naive build'}</div>
                <div className="muted">waiting …</div>
              </div>
            ),
          )}
        </div>
      )
    case 'build': {
      const rows = ['unfused', 'fused'].flatMap((tag) =>
        Object.entries(data[tag]?.ptxas ?? {})
          .sort(([a], [b]) => a.localeCompare(b))
          .map(([kernel, s]) => ({ tag, kernel, ...s })),
      )
      if (rows.length === 0) return <p className="muted">waiting …</p>
      return (
        <table>
          <thead>
            <tr>
              <th>build</th>
              <th>kernel</th>
              <th className="num">registers</th>
              <th className="num">on-chip shared memory</th>
              <th className="num">spilled to slow memory</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={`${r.tag}-${r.kernel}`}>
                <td>{r.tag}</td>
                <td className="mono">{r.kernel}</td>
                <td className="num">{r.regs ?? '?'}</td>
                <td className="num">{r.smem ?? 0} B</td>
                <td className={`num ${r.spills ? 'bad' : 'good'}`}>{r.spills ?? 0} B</td>
              </tr>
            ))}
          </tbody>
        </table>
      )
    }
    case 'execute': {
      const any = data.unfused || data.fused
      if (!any) return <p className="muted">waiting …</p>
      return (
        <div className="emit-grid">
          {['unfused', 'fused'].map((tag) =>
            data[tag] ? (
              <div className="emit-card" key={tag}>
                <div className="emit-title">
                  {tag === 'fused' ? 'fused build' : 'naive build'} — {ms(data[tag].total_ms)} ms
                </div>
                <table>
                  <tbody>
                    {Object.entries(data[tag].regions).map(([region, value]) => (
                      <tr key={region}>
                        <td className="mono">{region}</td>
                        <td className="num">{ms(value)} ms</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="emit-card" key={tag}>
                <div className="emit-title">{tag === 'fused' ? 'fused build' : 'naive build'}</div>
                <div className="muted">waiting …</div>
              </div>
            ),
          )}
        </div>
      )
    }
    case 'validate': {
      const rows = []
      for (const tag of ['unfused', 'fused']) {
        const v = data[tag]
        if (!v) continue
        rows.push({
          label: tag === 'fused' ? 'GPU — fused build' : 'GPU — naive build',
          checksum: v.checksum,
          preview: v.preview,
          verdict: `${v.correctness.ok ? 'PASS' : 'FAIL'} (max abs ${v.correctness.max_abs.toExponential(2)})`,
          ok: v.correctness.ok,
        })
      }
      if (data.oracle) {
        rows.push({
          label: `reference — ${data.oracle.precision}`,
          checksum: data.oracle.checksum,
          preview: data.oracle.preview,
          verdict: 'the yardstick',
          ok: true,
        })
      }
      if (rows.length === 0) return <p className="muted">waiting …</p>
      return (
        <table>
          <thead>
            <tr>
              <th>result</th>
              <th>checksum</th>
              <th>first output values</th>
              <th>matches reference?</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.label}>
                <td>{r.label}</td>
                <td className="num">{r.checksum}</td>
                <td className="mono small">
                  [{r.preview?.map((v) => v.toFixed(4)).join(', ')} …]
                </td>
                <td className={r.ok ? 'good' : 'bad'}>{r.verdict}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )
    }
    default:
      return null
  }
}

function StageCard({ meta, step, entry, workload, onOpenFile }) {
  const status = entry?.status ?? 'pending'
  return (
    <section className={`stage ${status}`}>
      <div className="stage-head">
        <span className="stage-num">{step}</span>
        <h3>{meta.title}</h3>
        <span className={`status ${status}`}>{status}</span>
      </div>
      <p className="stage-explain">{meta.explain}</p>
      {entry?.data?.error && <p className="error">{entry.data.error}</p>}
      <div className="stage-body">
        <StageBody id={meta.id} data={entry?.data} workload={workload} onOpenFile={onOpenFile} />
      </div>
    </section>
  )
}

function Recorded({ data }) {
  return (
    <section className="card">
      <h2>recorded evaluation — docs/REPORT.md</h2>
      <p className="muted">
        {data.environment.iterations} · {data.environment.gpu}
      </p>
      <table>
        <thead>
          <tr>
            <th>workload</th>
            <th className="num">fused ms</th>
            <th className="num">no-fusion ms</th>
            <th className="num">speedup</th>
          </tr>
        </thead>
        <tbody>
          {data.workloads.map((w) => {
            const f = w.variants.fused
            const u = w.variants.unfused
            return (
              <tr key={w.workload}>
                <td>{w.workload}</td>
                <td className="num">{ms(f.total_ms)}</td>
                <td className="num">{ms(u.total_ms)}</td>
                <td className="num good">{big(u.total_ms / f.total_ms)}×</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="muted legend">
        the full numbers — correctness, ptxas, per-region verdicts, rejected-fusion validation —
        live in docs/REPORT.md and results/results.json
      </p>
    </section>
  )
}

function App() {
  const [workloads, setWorkloads] = useState([])
  const [selected, setSelected] = useState(null)
  const [iters, setIters] = useState(150)
  const [warmup, setWarmup] = useState(50)
  const [jobId, setJobId] = useState(null)
  const [job, setJob] = useState(null)
  const [viewer, setViewer] = useState(null)
  const [recorded, setRecorded] = useState(null)
  const sinceRef = useRef(0)
  const logRef = useRef(null)

  useEffect(() => {
    getJSON('/api/workloads')
      .then((ws) => {
        setWorkloads(ws)
        setSelected((current) => current ?? ws[0]?.id ?? null)
      })
      .catch((e) => setJob({ status: 'failed', log: [], error: e.message, progress: 0, total_steps: 9 }))
    getJSON('/api/recorded').then(setRecorded).catch(() => {})
  }, [])

  useEffect(() => {
    if (!jobId) return undefined
    let stopped = false
    let timer = null
    const tick = async () => {
      try {
        const data = await getJSON(`/api/jobs/${jobId}?since=${sinceRef.current}`)
        sinceRef.current = data.next
        setJob((prev) => ({ ...data, log: [...(prev?.log ?? []), ...data.log] }))
        if (data.status === 'done' || data.status === 'failed') return
      } catch {
        /* transient failure -- keep polling */
      }
      if (!stopped) timer = setTimeout(tick, 900)
    }
    tick()
    return () => {
      stopped = true
      clearTimeout(timer)
    }
  }, [jobId])

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight
  }, [job?.log])

  const running = job?.status === 'queued' || job?.status === 'running'
  const selectedWorkload = workloads.find((w) => w.id === selected) ?? null

  const stageEntries = useMemo(() => {
    const map = Object.fromEntries((job?.stages ?? []).map((s) => [s.id, s]))
    if (!job && selectedWorkload) {
      map.input = {
        id: 'input',
        status: 'pending',
        data: {
          source: selectedWorkload.source,
          consts: selectedWorkload.consts,
          tensors: selectedWorkload.tensors,
          dsl: selectedWorkload.dsl,
          seed: '—',
        },
      }
    }
    return map
  }, [job, selectedWorkload])

  const startRun = async () => {
    setViewer(null)
    setJob(null)
    sinceRef.current = 0
    setJobId(null)
    try {
      const { job_id } = await post('/api/run', { workload: selected, iters, warmup })
      setJobId(job_id)
    } catch (e) {
      setJob({ status: 'failed', log: [], error: e.message, progress: 0, total_steps: 9, stages: [] })
    }
  }

  const openFile = async (title, url, pretty = false) => {
    setViewer({ title, text: 'loading ...' })
    try {
      const text = await fetch(url).then((r) => r.text())
      setViewer({ title, text: pretty ? JSON.stringify(JSON.parse(text), null, 2) : text })
    } catch (e) {
      setViewer({ title, text: `error: ${e.message}` })
    }
  }

  return (
    <div className="shell">
      <header>
        <div>
          <h1>
            FusionC <span className="thin">· live compiler dashboard</span>
          </h1>
          <p className="muted">
            a tensor-pipeline program goes in; fused GPU code comes out. Every step below is
            shown as it happens: text → tokens → tree → graph → fusion decisions → CUDA C →
            compiled GPU program → timings → correctness check.
          </p>
        </div>
        {recorded && (
          <div className="env">
            <span>{recorded.environment.gpu.split(',')[0]}</span>
            <span>CUDA {recorded.environment.nvcc.match(/release ([\d.]+)/)?.[1] ?? '11.x'}</span>
          </div>
        )}
      </header>

      <main>
        <section className="card">
          <h2>1 · pick a workload</h2>
          <div className="workloads">
            {workloads.map((w) => (
              <WorkloadCard
                key={w.id}
                w={w}
                selected={w.id === selected}
                onSelect={setSelected}
                disabled={running}
              />
            ))}
          </div>

          <h2>2 · run the whole pipeline</h2>
          <div className="controls">
            <label>
              timed iterations
              <input
                type="number"
                min="10"
                max="2000"
                value={iters}
                disabled={running}
                onChange={(e) => setIters(Number(e.target.value) || 10)}
              />
            </label>
            <label>
              warm-up
              <input
                type="number"
                min="0"
                max="500"
                value={warmup}
                disabled={running}
                onChange={(e) => setWarmup(Number(e.target.value) || 0)}
              />
            </label>
            <button className="run" onClick={startRun} disabled={running || !selected}>
              {running ? 'running …' : 'Run: fused vs naive, live'}
            </button>
          </div>

          {job && (
            <div className="progress">
              <div className="progress-track">
                <div
                  className="progress-fill"
                  style={{
                    width: `${Math.round(((job.progress ?? 0) / (job.total_steps || 9)) * 100)}%`,
                  }}
                />
              </div>
              <span className="muted">
                step {job.progress ?? 0}/{job.total_steps ?? 9} · {job.status}
              </span>
            </div>
          )}
          {job?.error && <p className="error">{job.error}</p>}
          {job && job.log.length > 0 && (
            <pre className="log" ref={logRef}>
              {job.log.join('\n')}
            </pre>
          )}
        </section>

        {STAGES.map((meta, i) => (
          <StageCard
            key={meta.id}
            meta={meta}
            step={i + 1}
            entry={stageEntries[meta.id]}
            workload={selected}
            onOpenFile={openFile}
          />
        ))}

        {job?.result && (
          <section className="card race">
            <h2>the race — same program, two builds</h2>
            <p className="muted">
              measured just now on {job.result.device} · {job.result.iters} timed iterations · seed{' '}
              {job.result.seed}
            </p>
            <div className="stats">
              <div className="stat hero">
                <span className="stat-label">speedup</span>
                <span className="stat-big">{big(job.result.totals.speedup)}×</span>
              </div>
              <div className="stat">
                <span className="stat-label">fused build</span>
                <span className="stat-big">{ms(job.result.totals.fused_ms)} ms</span>
                <span className="muted">{job.result.kernels.fused} kernels</span>
              </div>
              <div className="stat">
                <span className="stat-label">naive build</span>
                <span className="stat-big">{ms(job.result.totals.unfused_ms)} ms</span>
                <span className="muted">{job.result.kernels.unfused} kernels</span>
              </div>
            </div>
            <RegionBars regions={job.result.regions} />
            <p className="muted legend">
              each fused region against the sum of the naive kernels it replaced · improved &gt;
              +5%, degraded &lt; −5%
            </p>
          </section>
        )}

        {recorded && <Recorded data={recorded} />}
      </main>

      {viewer && (
        <div className="viewer" onClick={() => setViewer(null)} role="presentation">
          <div className="viewer-panel" onClick={(e) => e.stopPropagation()} role="presentation">
            <div className="viewer-head">
              <span>{viewer.title}</span>
              <button onClick={() => setViewer(null)}>close</button>
            </div>
            <pre>{viewer.text}</pre>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
