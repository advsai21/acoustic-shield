import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import {
  Radio,
  PhoneCall,
  PhoneOff,
  UserCheck,
  UserX,
  ShieldCheck,
  ShieldAlert,
  ShieldX,
  Check,
  ArrowLeftRight,
  Mic,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Fingerprint,
  AlertTriangle,
  Power,
} from "lucide-react";

/* ---------------------------------------------------------------
   Voice Integrity Verification — Dashboard / Alert UI (Role 5)
   Covers SRS FR-8 + the pipeline doc's demo script:
     Step 1 — Enroll a reference voiceprint
     Step 2 — Contextual signals (transcript + call metadata)
     Step 3 — Run the pipeline (live gauge, factor breakdown,
              recommended action, JSON record)

   MOCK MODE: this file simulates the backend locally so the UI can
   be built/demoed before the real FastAPI service exists. Flip
   POLL_MODE to "live" and point API_BASE at the real backend —
   fetchScoreFromApi() / enrollOnApi() already match the documented
   contracts (POST /enroll, GET /score/{session_id}).
----------------------------------------------------------------- */

const POLL_MODE = "mock"; // "mock" | "live"
const API_BASE = "http://localhost:8000";
const POLL_INTERVAL_MS = 1500;

// The real pipeline runs two different cadences: spoof/prosody/context
// score per audio chunk (~0.3–1s), while speaker verification only
// re-scores once its rolling buffer fills (~1.8s) — see README's
// "Important gotcha" section. The mock timers below reproduce that gap
// on purpose, so the UI's freshness badge has something real to show.
const CHUNK_INTERVAL_MS = 500;
const BUFFER_INTERVAL_MS = 1800;

const COLORS = {
  bg: "#0B0F17",
  panel: "#121826",
  panelAlt: "#171F2E",
  border: "#26324A",
  borderSoft: "#1D2637",
  text: "#E7ECF3",
  textDim: "#8695AC",
  textFaint: "#576079",
  accent: "#35C0C7",
  low: "#3DD68C",
  medium: "#F2B84B",
  high: "#FF5A5A",
};

const BANDS = {
  low: { key: "low", label: "Low", color: COLORS.low, action: "Allow" },
  medium: { key: "medium", label: "Medium", color: COLORS.medium, action: "Callback Verification" },
  high: { key: "high", label: "High", color: COLORS.high, action: "Require MFA Re-verification" },
  critical: { key: "high", label: "High", color: COLORS.high, action: "Escalate to Supervisor" },
};

const FACTOR_META = [
  { key: "spoof", label: "Spoof / synthetic likelihood", invert: false },
  { key: "speaker", label: "Speaker match to voiceprint", invert: true },
  { key: "prosody", label: "Prosody anomaly", invert: false },
  { key: "context", label: "Contextual risk", invert: false },
];

// Audio-derived targets — driven by the demo "inject sample" buttons.
// Context risk is driven separately, by the transcript + metadata panel.
const AUDIO_TARGETS = {
  idle: { spoof: 0.04, speaker: 0.0, prosody: 0.04 },
  genuine: { spoof: 0.08, speaker: 0.93, prosody: 0.1 },
  spoofed: { spoof: 0.86, speaker: 0.27, prosody: 0.79 },
};

const WEIGHTS = { spoof: 0.4, speaker: 0.3, prosody: 0.15, context: 0.15 };

// Spoof detection is treated as the core signal per the SRS's Reliability
// NFR ("graceful degradation — if one module fails, core spoof-detection
// score still functions"), so it's the one module that can't be knocked
// offline in the demo. The other three can be toggled to rehearse degraded
// mode: their weight gets redistributed across whatever's still active.
const DEGRADABLE_MODULES = [
  { key: "speaker", owner: "Role 2", name: "Speaker Consistency Verifier" },
  { key: "prosody", owner: "Role 2", name: "Prosody & Behavioral Feature Extractor" },
  { key: "context", owner: "Role 4", name: "Contextual Signal Engine" },
];

const RISK_PHRASES = [
  "otp",
  "one time password",
  "one-time code",
  "verification code",
  "pin number",
  "password",
  "wire transfer",
  "gift card",
  "urgent",
  "social security",
  "bank details",
  "verify your account",
  "suspended account",
  "act now",
];

const METADATA_FLAGS = [
  { key: "unknownCaller", label: "Unknown / unrecognized caller ID", weight: 0.16 },
  { key: "spoofedCallerId", label: "Caller ID appears spoofed", weight: 0.2 },
  { key: "unusualTime", label: "Unusual call time (late night)", weight: 0.1 },
  { key: "geoAnomaly", label: "Geographic anomaly vs. usual pattern", weight: 0.14 },
  { key: "firstContact", label: "First-time contact with this number", weight: 0.08 },
];

function computeContextScore(transcript, flags) {
  const t = transcript.toLowerCase();
  let phraseScore = 0;
  RISK_PHRASES.forEach((p) => {
    if (t.includes(p)) phraseScore += 0.22;
  });
  phraseScore = Math.min(0.7, phraseScore);

  let metaScore = 0;
  METADATA_FLAGS.forEach((m) => {
    if (flags[m.key]) metaScore += m.weight;
  });

  return Math.max(0, Math.min(1, phraseScore + metaScore));
}

function computeScore(f, moduleDown = {}) {
  const activeKeys = Object.keys(WEIGHTS).filter((k) => k === "spoof" || !moduleDown[k]);
  const totalWeight = activeKeys.reduce((sum, k) => sum + WEIGHTS[k], 0);
  const contribution = { spoof: f.spoof, speaker: 1 - f.speaker, prosody: f.prosody, context: f.context };
  const raw = activeKeys.reduce((sum, k) => sum + (WEIGHTS[k] / totalWeight) * contribution[k], 0);
  return Math.round(Math.max(0, Math.min(1, raw)) * 100);
}

function bandForScore(score) {
  if (score <= 30) return BANDS.low;
  if (score <= 65) return BANDS.medium;
  if (score <= 85) return BANDS.high;
  return BANDS.critical;
}

function ease(prev, target, pull, jitter) {
  const next = { ...prev };
  for (const k of Object.keys(target)) {
    const moved = prev[k] + (target[k] - prev[k]) * pull;
    const noisy = moved + (Math.random() - 0.5) * jitter;
    next[k] = Math.max(0, Math.min(1, noisy));
  }
  return next;
}

function timeNow() {
  return new Date().toLocaleTimeString("en-US", { hour12: false });
}

function randomId(prefix) {
  return `${prefix}_${Math.random().toString(36).slice(2, 8)}`;
}

// Placeholders for the real integration — shaped to match the SRS's
// documented API contract so swapping POLL_MODE to "live" is close
// to a drop-in change.
async function fetchScoreFromApi(sessionId) {
  const res = await fetch(`${API_BASE}/score/${sessionId}`);
  if (!res.ok) throw new Error(`score fetch failed: ${res.status}`);
  return res.json();
}

async function enrollOnApi(sampleBlob) {
  const form = new FormData();
  form.append("sample", sampleBlob);
  const res = await fetch(`${API_BASE}/enroll`, { method: "POST", body: form });
  if (!res.ok) throw new Error(`enroll failed: ${res.status}`);
  return res.json();
}

// Tells the mock/real backend which scenario to simulate for this session —
// pairs with the /analyze endpoint in mock_backend.py.
async function analyzeOnApi(sessionId, speakerId, scenario) {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, speaker_id: speakerId, scenario }),
  });
  if (!res.ok) throw new Error(`analyze failed: ${res.status}`);
  return res.json();
}

function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, endAngle);
  const end = polarToCartesian(cx, cy, r, startAngle);
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1";
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`;
}

const GAUGE_START = -130;
const GAUGE_END = 130;
const GAUGE_SPAN = GAUGE_END - GAUGE_START;

function angleForScore(score) {
  return GAUGE_START + (score / 100) * GAUGE_SPAN;
}

function RiskGauge({ score, bandColor, history }) {
  const cx = 130,
    cy = 130,
    r = 100;
  const zones = [
    { from: 0, to: 30, color: COLORS.low },
    { from: 30, to: 65, color: COLORS.medium },
    { from: 65, to: 100, color: COLORS.high },
  ];

  const sparkW = 148,
    sparkH = 34;
  const points =
    history.length > 1
      ? history
          .map((h, i) => {
            const x = (i / (history.length - 1)) * sparkW;
            const y = sparkH - (h / 100) * sparkH;
            return `${x.toFixed(1)},${y.toFixed(1)}`;
          })
          .join(" ")
      : "";

  return (
    <div className="flex flex-col items-center">
      <svg width={260} height={190} viewBox="0 0 260 190">
        {zones.map((z) => (
          <path
            key={z.from}
            d={describeArc(cx, cy, r, angleForScore(z.from), angleForScore(z.to))}
            fill="none"
            stroke={z.color}
            strokeOpacity={0.16}
            strokeWidth={14}
            strokeLinecap="butt"
          />
        ))}
        <path
          d={describeArc(cx, cy, r, GAUGE_START, angleForScore(score))}
          fill="none"
          stroke={bandColor}
          strokeWidth={14}
          strokeLinecap="round"
          style={{ transition: "d 0.4s ease, stroke 0.4s ease" }}
        />
        {[0, 30, 65, 100].map((tick) => {
          const p = polarToCartesian(cx, cy, r + 20, angleForScore(tick));
          return (
            <text
              key={tick}
              x={p.x}
              y={p.y}
              fill={COLORS.textFaint}
              fontSize="10"
              fontFamily="IBM Plex Mono, monospace"
              textAnchor="middle"
              dominantBaseline="middle"
            >
              {tick}
            </text>
          );
        })}
        <text
          x={cx}
          y={cy - 6}
          fill={COLORS.text}
          fontSize="46"
          fontWeight="600"
          fontFamily="IBM Plex Mono, monospace"
          textAnchor="middle"
        >
          {score}
        </text>
        <text
          x={cx}
          y={cy + 20}
          fill={COLORS.textDim}
          fontSize="11"
          fontFamily="IBM Plex Sans, sans-serif"
          textAnchor="middle"
          letterSpacing="0.5"
        >
          / 100
        </text>
      </svg>
      <div style={{ color: COLORS.textDim, fontSize: 12, marginTop: -6, fontFamily: "IBM Plex Sans, sans-serif" }}>
        Impersonation Risk Score
      </div>
      <svg width={sparkW} height={sparkH} style={{ marginTop: 10 }}>
        {points && (
          <polyline points={points} fill="none" stroke={bandColor} strokeWidth={1.6} strokeLinejoin="round" strokeLinecap="round" />
        )}
      </svg>
    </div>
  );
}

function FactorBar({ label, value, invert, cadenceNote, flash, offline }) {
  const pct = Math.round(value * 100);
  const goodness = invert ? value : 1 - value;
  const color = goodness > 0.66 ? COLORS.low : goodness > 0.35 ? COLORS.medium : COLORS.high;

  return (
    <div style={{ marginBottom: 14, opacity: offline ? 0.5 : 1 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
        <span className="flex items-center gap-1.5">
          <span style={{ color: COLORS.textDim, fontSize: 12.5, fontFamily: "IBM Plex Sans, sans-serif" }}>{label}</span>
          {flash !== undefined && !offline && (
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: flash ? COLORS.accent : COLORS.borderSoft,
                transition: "background 0.25s ease",
                flexShrink: 0,
              }}
            />
          )}
        </span>
        <span
          style={{
            color: offline ? COLORS.textFaint : color,
            fontSize: 12.5,
            fontFamily: "IBM Plex Mono, monospace",
            fontWeight: 600,
          }}
        >
          {offline ? "—" : `${pct}%`}
        </span>
      </div>
      <div style={{ height: 8, borderRadius: 4, background: COLORS.borderSoft, overflow: "hidden" }}>
        {offline ? (
          <div
            style={{
              width: "100%",
              height: "100%",
              backgroundImage: `repeating-linear-gradient(135deg, ${COLORS.borderSoft}, ${COLORS.borderSoft} 4px, ${COLORS.panelAlt} 4px, ${COLORS.panelAlt} 8px)`,
            }}
          />
        ) : (
          <div
            style={{
              width: `${pct}%`,
              height: "100%",
              background: color,
              borderRadius: 4,
              transition: "width 0.4s ease, background 0.4s ease",
            }}
          />
        )}
      </div>
      {cadenceNote && <div style={{ fontSize: 10.5, color: COLORS.textFaint, marginTop: 4 }}>{cadenceNote}</div>}
    </div>
  );
}

function DemoButton({ icon: Icon, label, onClick, disabled, tone }) {
  const toneColor = tone === "danger" ? COLORS.high : tone === "success" ? COLORS.low : COLORS.accent;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex items-center gap-2 px-3 py-2 rounded-md text-sm"
      style={{
        fontFamily: "IBM Plex Sans, sans-serif",
        background: disabled ? COLORS.panelAlt : "#1B2436",
        color: disabled ? COLORS.textFaint : COLORS.text,
        border: `1px solid ${disabled ? COLORS.borderSoft : COLORS.border}`,
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.55 : 1,
        transition: "border-color 0.15s ease",
      }}
      onMouseEnter={(e) => {
        if (!disabled) e.currentTarget.style.borderColor = toneColor;
      }}
      onMouseLeave={(e) => {
        if (!disabled) e.currentTarget.style.borderColor = COLORS.border;
      }}
    >
      <Icon size={15} color={disabled ? COLORS.textFaint : toneColor} />
      {label}
    </button>
  );
}

function Checkbox({ checked, onChange, label }) {
  return (
    <label className="flex items-center gap-2" style={{ cursor: "pointer", fontSize: 12.5 }}>
      <span
        onClick={onChange}
        style={{
          width: 16,
          height: 16,
          borderRadius: 4,
          border: `1.5px solid ${checked ? COLORS.accent : COLORS.border}`,
          background: checked ? COLORS.accent : "transparent",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          transition: "all 0.15s ease",
        }}
      >
        {checked && <Check size={11} color="#0B0F17" strokeWidth={3} />}
      </span>
      <span style={{ color: checked ? COLORS.text : COLORS.textDim }}>{label}</span>
    </label>
  );
}

export default function VoiceIntegrityDashboard() {
  const [sessionId, setSessionId] = useState(() => randomId("sess"));
  const [voiceprintId, setVoiceprintId] = useState(null);
  const [enrolling, setEnrolling] = useState(false);

  const [callState, setCallState] = useState("idle"); // idle | live
  const [audioTarget, setAudioTarget] = useState("idle"); // idle | genuine | spoofed

  const [transcript, setTranscript] = useState("");
  const [flags, setFlags] = useState({
    unknownCaller: false,
    spoofedCallerId: false,
    unusualTime: false,
    geoAnomaly: false,
    firstContact: false,
  });

  const [factors, setFactors] = useState({ spoof: 0.04, speaker: 0.0, prosody: 0.04, context: 0 });
  const [score, setScore] = useState(0);
  const [history, setHistory] = useState([0]);
  const [log, setLog] = useState([]);
  const [showJson, setShowJson] = useState(false);
  const [lastDecision, setLastDecision] = useState(null);
  const [speakerFlash, setSpeakerFlash] = useState(false);
  const [lastSpeakerUpdate, setLastSpeakerUpdate] = useState(null);
  const [clockTick, setClockTick] = useState(0); // forces re-render so "Ns ago" ticks
  const [moduleDown, setModuleDown] = useState({ speaker: false, prosody: false, context: false });

  const band = bandForScore(score);
  const lastBandRef = useRef(band.key);
  const prevSpeakerRef = useRef(factors.speaker);

  const addLog = useCallback((text) => {
    setLog((prev) => [{ time: timeNow(), text }, ...prev].slice(0, 8));
  }, []);

  // Contextual risk reacts immediately to transcript + metadata, independent
  // of the audio-scenario simulation below.
  const contextScore = useMemo(() => computeContextScore(transcript, flags), [transcript, flags]);
  useEffect(() => {
    setFactors((prev) => ({ ...prev, context: contextScore }));
  }, [contextScore]);

  // Recompute the fused score whenever any factor or module's up/down
  // state changes, and detect when speaker_similarity actually moved
  // (vs. just being re-rendered) — that's what drives the freshness
  // badge, independent of which mode or timer produced the change.
  useEffect(() => {
    const s = computeScore(factors, moduleDown);
    setScore(s);
    if (callState === "live") {
      setHistory((h) => [...h, s].slice(-24));
    }
    if (!moduleDown.speaker && Math.abs(factors.speaker - prevSpeakerRef.current) > 0.0015) {
      prevSpeakerRef.current = factors.speaker;
      setLastSpeakerUpdate(Date.now());
      setSpeakerFlash(true);
      setTimeout(() => setSpeakerFlash(false), 350);
    }
  }, [factors, callState, moduleDown]);

  // Chunk-cadence factors (spoof, prosody) — updates roughly per audio chunk.
  // A module that's simulated as offline just stops updating; its last
  // value is frozen and excluded from fusion by computeScore above.
  useEffect(() => {
    if (callState !== "live" || POLL_MODE === "live") return;
    const id = setInterval(() => {
      setFactors((prev) => {
        const target = { spoof: AUDIO_TARGETS[audioTarget].spoof };
        if (!moduleDown.prosody) target.prosody = AUDIO_TARGETS[audioTarget].prosody;
        return { ...prev, ...ease(prev, target, 0.35, 0.02) };
      });
    }, CHUNK_INTERVAL_MS);
    return () => clearInterval(id);
  }, [callState, audioTarget, moduleDown.prosody]);

  // Buffer-cadence factor (speaker) — only re-scores once the rolling
  // ~1.8s buffer fills, per the README's documented buffer fix.
  useEffect(() => {
    if (callState !== "live" || POLL_MODE === "live" || moduleDown.speaker) return;
    const id = setInterval(() => {
      setFactors((prev) => ({
        ...prev,
        ...ease(prev, { speaker: AUDIO_TARGETS[audioTarget].speaker }, 0.55, 0.015),
      }));
    }, BUFFER_INTERVAL_MS);
    return () => clearInterval(id);
  }, [callState, audioTarget, moduleDown.speaker]);

  // Live-backend polling covers all four factors from one endpoint —
  // the backend is responsible for its own internal buffering.
  useEffect(() => {
    if (callState !== "live" || POLL_MODE !== "live") return;
    const id = setInterval(async () => {
      try {
        const data = await fetchScoreFromApi(sessionId);
        setFactors((prev) => ({
          spoof: data.contributing_factors.spoof_score,
          speaker: data.contributing_factors.speaker_similarity,
          prosody: data.contributing_factors.prosody_anomaly_score,
          context: prev.context,
        }));
      } catch (e) {
        addLog("Score poll failed — check backend connection");
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [callState, sessionId, addLog]);

  // Ticks once a second while live so the "updated Ns ago" label stays current.
  useEffect(() => {
    if (callState !== "live") return;
    const id = setInterval(() => setClockTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [callState]);

  useEffect(() => {
    if (band.key !== lastBandRef.current) {
      lastBandRef.current = band.key;
      if (callState === "live") {
        addLog(`Risk band changed to ${band.label.toUpperCase()} — ${band.action}`);
        setLastDecision(null);
      }
    }
  }, [band, callState, addLog]);

  const enroll = async () => {
    setEnrolling(true);
    addLog("Enrolling reference voiceprint…");
    if (POLL_MODE === "live") {
      try {
        const data = await enrollOnApi(new Blob());
        setVoiceprintId(data.speaker_id);
        addLog(`Voiceprint enrolled — ${data.speaker_id}`);
      } catch (e) {
        addLog("Enrollment failed — check backend connection");
      }
      setEnrolling(false);
      return;
    }
    setTimeout(() => {
      const id = randomId("vp");
      setVoiceprintId(id);
      setEnrolling(false);
      addLog(`Voiceprint enrolled — ${id}`);
    }, 1100);
  };

  const resetEnrollment = () => {
    setVoiceprintId(null);
    addLog("Voiceprint cleared");
  };

  const startCall = () => {
    const newSessionId = randomId("sess");
    setSessionId(newSessionId);
    setCallState("live");
    setAudioTarget("genuine");
    setFactors((f) => ({ spoof: 0.04, speaker: 0.0, prosody: 0.04, context: f.context }));
    setHistory([0]);
    lastBandRef.current = "low";
    setLastDecision(null);
    prevSpeakerRef.current = 0;
    setLastSpeakerUpdate(null);
    addLog("Call started against enrolled voiceprint");
    if (POLL_MODE === "live") {
      analyzeOnApi(newSessionId, voiceprintId, "genuine").catch(() =>
        addLog("Analyze call failed — check backend connection")
      );
    }
  };

  const injectGenuine = () => {
    setAudioTarget("genuine");
    addLog("Injected genuine sample");
    if (POLL_MODE === "live") {
      analyzeOnApi(sessionId, voiceprintId, "genuine").catch(() => addLog("Analyze call failed"));
    }
  };

  const injectSpoofed = () => {
    setAudioTarget("spoofed");
    addLog("Injected cloned/TTS sample");
    if (POLL_MODE === "live") {
      analyzeOnApi(sessionId, voiceprintId, "spoofed").catch(() => addLog("Analyze call failed"));
    }
  };

  const endCall = () => {
    setCallState("idle");
    setAudioTarget("idle");
    setFactors((f) => ({ spoof: 0.04, speaker: 0.0, prosody: 0.04, context: f.context }));
    setHistory([0]);
    setLastDecision(null);
    addLog("Call ended — session reset");
  };

  const recordDecision = (decision) => {
    setLastDecision({ decision, time: timeNow(), action: band.action });
    addLog(decision === "accept" ? `Operator accepted: ${band.action}` : "Operator overrode recommendation to Allow");
  };

  const toggleModule = (key) => {
    setModuleDown((prev) => {
      const next = { ...prev, [key]: !prev[key] };
      const meta = DEGRADABLE_MODULES.find((m) => m.key === key);
      addLog(next[key] ? `${meta.name} went offline (simulated)` : `${meta.name} recovered`);
      return next;
    });
  };

  const downModules = DEGRADABLE_MODULES.filter((m) => moduleDown[m.key]);
  const isDegraded = downModules.length > 0;

  const BandIcon = band.key === "low" ? ShieldCheck : band.key === "medium" ? ShieldAlert : ShieldX;

  const jsonRecord = {
    session_id: sessionId,
    risk_score: score,
    risk_band: band.label.toLowerCase(),
    contributing_factors: {
      spoof_score: Number(factors.spoof.toFixed(3)),
      speaker_similarity: Number(factors.speaker.toFixed(3)),
      prosody_anomaly_score: Number(factors.prosody.toFixed(3)),
      context_risk_score: Number(factors.context.toFixed(3)),
    },
    recommended_action: band.action,
    operator_decision: lastDecision ? lastDecision.decision : null,
    module_status: {
      spoof: "active",
      speaker: moduleDown.speaker ? "offline" : "active",
      prosody: moduleDown.prosody ? "offline" : "active",
      context: moduleDown.context ? "offline" : "active",
    },
    timestamp: timeNow(),
  };

  return (
    <div style={{ background: COLORS.bg, color: COLORS.text, minHeight: "100%", fontFamily: "IBM Plex Sans, sans-serif", padding: "20px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
        @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.35; } }
        @keyframes eqbar { 0%,100% { transform: scaleY(0.4); } 50% { transform: scaleY(1); } }
        textarea::placeholder { color: ${COLORS.textFaint}; }
      `}</style>

      <div style={{ maxWidth: 1040, margin: "0 auto" }}>
        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-3" style={{ marginBottom: 18 }}>
          <div className="flex items-center gap-3">
            <div className="flex items-end gap-[2px]" style={{ height: 18 }}>
              {[0, 1, 2, 3].map((i) => (
                <div
                  key={i}
                  style={{
                    width: 3,
                    height: "100%",
                    background: callState === "live" ? COLORS.accent : COLORS.borderSoft,
                    borderRadius: 1,
                    transformOrigin: "bottom",
                    animation: callState === "live" ? `eqbar ${0.6 + i * 0.15}s ease-in-out infinite` : "none",
                  }}
                />
              ))}
            </div>
            <div>
              <div style={{ fontSize: 15, fontWeight: 600, letterSpacing: 0.2 }}>Voice Integrity Console</div>
              <div style={{ fontSize: 11.5, color: COLORS.textFaint, fontFamily: "IBM Plex Mono, monospace" }}>{sessionId}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Radio
              size={13}
              color={callState === "live" ? COLORS.low : COLORS.textFaint}
              style={{ animation: callState === "live" ? "pulse 1.4s ease-in-out infinite" : "none" }}
            />
            <span style={{ fontSize: 12.5, color: callState === "live" ? COLORS.low : COLORS.textFaint }}>
              {callState === "live" ? "Call in progress" : "No active call"}
            </span>
          </div>
        </div>

        {/* Step 1 — Enrollment */}
        <div
          className="flex items-center justify-between flex-wrap gap-3"
          style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "14px 18px", marginBottom: 16 }}
        >
          <div className="flex items-center gap-3">
            <Fingerprint size={18} color={voiceprintId ? COLORS.low : COLORS.textFaint} />
            <div>
              <div style={{ fontSize: 11.5, color: COLORS.textFaint }}>Step 1 — Reference voiceprint</div>
              <div style={{ fontSize: 13.5, fontFamily: voiceprintId ? "IBM Plex Mono, monospace" : "IBM Plex Sans, sans-serif" }}>
                {enrolling ? "Enrolling…" : voiceprintId ? voiceprintId : "Not enrolled"}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {!voiceprintId ? (
              <DemoButton icon={Mic} label={enrolling ? "Enrolling…" : "Enroll voiceprint"} onClick={enroll} disabled={enrolling} />
            ) : (
              <DemoButton icon={RotateCcw} label="Clear enrollment" onClick={resetEnrollment} disabled={callState === "live"} />
            )}
          </div>
        </div>

        {/* Gauge + Factors */}
        <div className="grid grid-cols-12 gap-4" style={{ marginBottom: 16 }}>
          <div
            className="col-span-12 md:col-span-5 flex items-center justify-center"
            style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "18px" }}
          >
            <RiskGauge score={score} bandColor={band.color} history={history} />
          </div>

          <div
            className="col-span-12 md:col-span-7"
            style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "20px" }}
          >
            <div className="flex items-center justify-between" style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 13, color: COLORS.textDim, fontWeight: 500 }}>Why this score?</div>
              {isDegraded && (
                <div style={{ fontSize: 11, color: COLORS.medium, fontFamily: "IBM Plex Mono, monospace" }}>
                  {4 - downModules.length} of 4 modules active
                </div>
              )}
            </div>
            {FACTOR_META.map((f) => {
              const offline = f.key !== "spoof" && moduleDown[f.key];
              if (f.key === "speaker") {
                const elapsed = lastSpeakerUpdate ? Math.max(0, Math.round((Date.now() - lastSpeakerUpdate) / 1000)) : null;
                const note = offline
                  ? "Module offline (simulated) — excluded from fused score"
                  : callState !== "live"
                  ? "Refreshes on a ~1.8s rolling buffer, not every chunk"
                  : elapsed === null
                  ? "Refreshes on a ~1.8s rolling buffer — waiting for first window"
                  : `Refreshes on a ~1.8s rolling buffer — updated ${elapsed}s ago`;
                return (
                  <FactorBar
                    key={f.key}
                    label={f.label}
                    value={factors[f.key]}
                    invert={f.invert}
                    flash={speakerFlash}
                    cadenceNote={note}
                    offline={offline}
                  />
                );
              }
              return (
                <FactorBar
                  key={f.key}
                  label={f.label}
                  value={factors[f.key]}
                  invert={f.invert}
                  offline={offline}
                  cadenceNote={offline ? "Module offline (simulated) — excluded from fused score" : undefined}
                />
              );
            })}
          </div>
        </div>

        {/* Step 2 — Contextual signals */}
        <div
          className="grid grid-cols-12 gap-4"
          style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "20px", marginBottom: 16 }}
        >
          <div className="col-span-12 md:col-span-7">
            <div style={{ fontSize: 13, color: COLORS.textDim, marginBottom: 10, fontWeight: 500 }}>Step 2 — Live transcript (contextual signal)</div>
            <textarea
              value={transcript}
              onChange={(e) => setTranscript(e.target.value)}
              placeholder='Try: "please share your OTP now" or "this is urgent, verify your account"'
              rows={3}
              style={{
                width: "100%",
                resize: "none",
                background: COLORS.panelAlt,
                border: `1px solid ${COLORS.border}`,
                borderRadius: 8,
                padding: "10px 12px",
                color: COLORS.text,
                fontSize: 13,
                fontFamily: "IBM Plex Sans, sans-serif",
                outline: "none",
              }}
            />
            <div style={{ marginTop: 8, fontSize: 11.5, color: COLORS.textFaint }}>
              Simulates ASR + keyword/intent spotting — flags OTP, credential, and urgency phrasing.
            </div>
          </div>
          <div className="col-span-12 md:col-span-5">
            <div style={{ fontSize: 13, color: COLORS.textDim, marginBottom: 10, fontWeight: 500 }}>Call metadata</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 9 }}>
              {METADATA_FLAGS.map((m) => (
                <Checkbox
                  key={m.key}
                  checked={flags[m.key]}
                  onChange={() => setFlags((prev) => ({ ...prev, [m.key]: !prev[m.key] }))}
                  label={m.label}
                />
              ))}
            </div>
            <div style={{ marginTop: 12, fontSize: 12, color: COLORS.textDim }}>
              Context risk contribution:{" "}
              <span style={{ fontFamily: "IBM Plex Mono, monospace", color: COLORS.text, fontWeight: 600 }}>
                {Math.round(contextScore * 100)}%
              </span>
            </div>
          </div>
        </div>

        {/* Degraded-mode notice */}
        {isDegraded && (
          <div
            className="flex items-center gap-2"
            style={{
              background: `${COLORS.medium}14`,
              border: `1px solid ${COLORS.medium}55`,
              borderRadius: 8,
              padding: "9px 14px",
              marginBottom: 12,
              fontSize: 12,
              color: COLORS.medium,
            }}
          >
            <AlertTriangle size={14} />
            Running in degraded mode — {downModules.map((m) => m.name).join(", ")} offline. Core spoof-detection score still functions.
          </div>
        )}

        {/* Action banner */}
        <div
          style={{
            background: COLORS.panel,
            border: `1px solid ${band.color}55`,
            borderLeft: `4px solid ${band.color}`,
            borderRadius: 10,
            padding: "16px 20px",
            marginBottom: 16,
          }}
          className="flex items-center justify-between flex-wrap gap-3"
        >
          <div className="flex items-center gap-3">
            <BandIcon size={22} color={band.color} />
            <div>
              <div style={{ fontSize: 11.5, color: COLORS.textFaint }}>{band.label} risk — recommended action</div>
              <div style={{ fontSize: 15.5, fontWeight: 600 }}>{band.action}</div>
              {lastDecision && (
                <div style={{ fontSize: 11.5, color: COLORS.textFaint, marginTop: 2 }}>
                  Operator {lastDecision.decision === "accept" ? "accepted" : "overrode"} at {lastDecision.time}
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => recordDecision("accept")}
              className="flex items-center gap-1.5 px-3 py-2 rounded-md text-sm"
              style={{ background: band.color, color: "#0B0F17", fontWeight: 600, border: "none", cursor: "pointer" }}
            >
              <Check size={14} /> Accept
            </button>
            <button
              onClick={() => recordDecision("override")}
              className="flex items-center gap-1.5 px-3 py-2 rounded-md text-sm"
              style={{ background: "transparent", color: COLORS.textDim, border: `1px solid ${COLORS.border}`, cursor: "pointer" }}
            >
              <ArrowLeftRight size={14} /> Override
            </button>
          </div>
        </div>

        {/* JSON breakdown */}
        <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 10, marginBottom: 16, overflow: "hidden" }}>
          <button
            onClick={() => setShowJson((v) => !v)}
            className="flex items-center justify-between w-full"
            style={{ padding: "12px 18px", background: "transparent", border: "none", cursor: "pointer" }}
          >
            <span style={{ fontSize: 11.5, color: COLORS.textFaint }}>Score record (API response shape)</span>
            {showJson ? <ChevronUp size={15} color={COLORS.textFaint} /> : <ChevronDown size={15} color={COLORS.textFaint} />}
          </button>
          {showJson && (
            <pre
              style={{
                margin: 0,
                padding: "0 18px 16px",
                fontSize: 12,
                lineHeight: 1.6,
                color: COLORS.textDim,
                fontFamily: "IBM Plex Mono, monospace",
                overflowX: "auto",
              }}
            >
              {JSON.stringify(jsonRecord, null, 2)}
            </pre>
          )}
        </div>

        {/* Presenter / demo controls */}
        <div style={{ border: `1px dashed ${COLORS.border}`, borderRadius: 10, padding: "14px 18px", marginBottom: 16 }}>
          <div style={{ fontSize: 11, color: COLORS.textFaint, marginBottom: 10, letterSpacing: 0.3 }}>
            Presenter controls — not part of the production dashboard
          </div>
          <div className="flex items-center gap-2 flex-wrap" style={{ marginBottom: 14 }}>
            <DemoButton icon={PhoneCall} label="Start call" onClick={startCall} disabled={callState === "live" || !voiceprintId} />
            <DemoButton icon={UserCheck} label="Inject genuine sample" onClick={injectGenuine} disabled={callState !== "live"} tone="success" />
            <DemoButton icon={UserX} label="Inject spoofed sample" onClick={injectSpoofed} disabled={callState !== "live"} tone="danger" />
            <DemoButton icon={PhoneOff} label="End call" onClick={endCall} disabled={callState !== "live"} />
          </div>
          {!voiceprintId && <div style={{ fontSize: 11.5, color: COLORS.textFaint, marginBottom: 14 }}>Enroll a voiceprint above before starting a call.</div>}

          <div style={{ fontSize: 11, color: COLORS.textFaint, marginBottom: 10 }}>
            Simulate module failure (Reliability NFR — spoof detection can't be knocked offline)
          </div>
          <div className="flex items-center gap-2 flex-wrap">
            {DEGRADABLE_MODULES.map((m) => (
              <DemoButton
                key={m.key}
                icon={moduleDown[m.key] ? Power : AlertTriangle}
                label={moduleDown[m.key] ? `Restore ${m.name}` : `Fail ${m.name}`}
                onClick={() => toggleModule(m.key)}
                tone={moduleDown[m.key] ? "success" : "danger"}
              />
            ))}
          </div>
        </div>

        {/* Activity log */}
        <div style={{ background: COLORS.panel, border: `1px solid ${COLORS.border}`, borderRadius: 10, padding: "14px 18px" }}>
          <div style={{ fontSize: 11.5, color: COLORS.textFaint, marginBottom: 10 }}>Session log</div>
          {log.length === 0 ? (
            <div style={{ fontSize: 12.5, color: COLORS.textFaint }}>No events yet.</div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {log.map((entry, i) => (
                <div key={i} className="flex gap-3" style={{ fontSize: 12.5 }}>
                  <span style={{ color: COLORS.textFaint, fontFamily: "IBM Plex Mono, monospace", flexShrink: 0 }}>{entry.time}</span>
                  <span style={{ color: COLORS.textDim }}>{entry.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
