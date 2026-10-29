import { useEffect, useRef, useState } from "react";
import {
  attachTask,
  getVersion,
  listTasks,
  upgradeEvents,
  upgradeOllama,
  type TaskEvent,
} from "../api";
import type { VersionInfo } from "../types";

export default function UpdateBanner({ ollamaUrl }: { ollamaUrl: string }) {
  const [info, setInfo] = useState<VersionInfo | null>(null);
  const [upgrading, setUpgrading] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [done, setDone] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [upgradedTo, setUpgradedTo] = useState<string | null>(null);
  const [password, setPassword] = useState("");
  const logRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    let alive = true;
    getVersion(ollamaUrl)
      .then((v) => alive && setInfo(v))
      .catch(() => alive && setInfo(null));
    return () => {
      alive = false;
    };
  }, [ollamaUrl]);

  // Keep the newest output in view.
  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [log]);

  /** One reader for an upgrade started here and one picked up after a reload. */
  const reader = (): ((ev: TaskEvent) => void) =>
    upgradeEvents(
      (text, replace) =>
        setLog((prev) =>
          // The download's progress bar redraws itself; show its latest state.
          replace && prev.length ? [...prev.slice(0, -1), text] : [...prev, text]
        ),
      (version) => setUpgradedTo(version)
    );

  async function run(follow: (onEvent: (ev: TaskEvent) => void) => Promise<void>) {
    setUpgrading(true);
    setDone(false);
    setError(null);
    setUpgradedTo(null);
    try {
      await follow(reader());
      setDone(true);
      // The installer restarted Ollama: say what's running now, and drop the card
      // once there is nothing left to upgrade to.
      setInfo(await getVersion(ollamaUrl).catch(() => null));
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setUpgrading(false);
      setPassword("");
    }
  }

  // An upgrade keeps running on the backend through a reload; pick it back up.
  useEffect(() => {
    let cancelled = false;
    listTasks()
      .then((tasks) => {
        const task = tasks.find((t) => t.kind === "ollama-upgrade");
        if (task && !cancelled) {
          setLog([]);
          run((onEvent) => attachTask(task.id, onEvent));
        }
      })
      .catch(() => {
        /* backend unreachable; nothing to resume */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function start() {
    setLog([]);
    run((onEvent) =>
      upgradeOllama(ollamaUrl, info?.needs_password ? password : null, onEvent)
    );
  }

  const started = upgrading || done || log.length > 0;
  if (!started && (!info || !info.update_available)) return null;
  const askPassword = !!info?.needs_password && !upgrading;

  return (
    <div className="update-card">
      <div className="update-head">
        {done ? "✓ Ollama upgraded" : "⬆️ Ollama update available"}
        {info && !done && (
          <div className="update-versions">
            <span className="ver old">{info.installed}</span>
            <span>→</span>
            <span className="ver new">{info.latest}</span>
          </div>
        )}
      </div>

      {info && !info.is_local ? (
        <div className="muted small">Ollama is remote — upgrade it on its host.</div>
      ) : (
        <>
          {!upgrading && !done && (
            <form
              className="upgrade-form"
              onSubmit={(e) => {
                e.preventDefault();
                start();
              }}
            >
              {askPassword && (
                <>
                  <input
                    type="password"
                    autoComplete="off"
                    placeholder="Your sudo password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                  <div className="muted small">
                    The installer replaces Ollama in /usr/local and restarts its
                    service, so it needs root. Your password goes only to sudo on this
                    machine, for this upgrade, and isn't saved.
                  </div>
                </>
              )}
              <div className="muted small">
                Ollama restarts at the end, which interrupts any answer still being
                written.
              </div>
              <button
                type="submit"
                className="btn primary block"
                disabled={askPassword && !password}
              >
                {error ? "Try again" : "Upgrade now"}
              </button>
            </form>
          )}
          {upgrading && <div className="muted small">Upgrading…</div>}
          {log.length > 0 && (
            <pre className="upgrade-log" ref={logRef}>
              {log.join("\n")}
            </pre>
          )}
          {error && <div className="small upgrade-error">✗ {error}</div>}
          {done && (
            <div className="muted small">
              {upgradedTo
                ? `Ollama ${upgradedTo} is running.`
                : "Installed. Restart the Ollama service if it doesn't come back on its own."}
            </div>
          )}
        </>
      )}
    </div>
  );
}
