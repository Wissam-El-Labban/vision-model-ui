import { useEffect, useState } from "react";
import { getVersion, upgradeOllama } from "../api";
import type { VersionInfo } from "../types";

export default function UpdateBanner({ ollamaUrl }: { ollamaUrl: string }) {
  const [info, setInfo] = useState<VersionInfo | null>(null);
  const [upgrading, setUpgrading] = useState(false);
  const [log, setLog] = useState<string[]>([]);
  const [done, setDone] = useState(false);

  useEffect(() => {
    let alive = true;
    getVersion(ollamaUrl)
      .then((v) => alive && setInfo(v))
      .catch(() => alive && setInfo(null));
    return () => {
      alive = false;
    };
  }, [ollamaUrl]);

  if (!info || !info.update_available) return null;

  async function upgrade() {
    setUpgrading(true);
    setLog([]);
    setDone(false);
    try {
      await upgradeOllama(ollamaUrl, (line) =>
        setLog((prev) => [...prev, line])
      );
    } catch (e) {
      setLog((prev) => [...prev, `Error: ${(e as Error).message}`]);
    } finally {
      setUpgrading(false);
      setDone(true);
    }
  }

  return (
    <div className="update-card">
      <div className="update-head">
        ⬆️ Ollama update available
        <div className="update-versions">
          <span className="ver old">{info.installed}</span>
          <span>→</span>
          <span className="ver new">{info.latest}</span>
        </div>
      </div>

      {info.is_local ? (
        !upgrading && !done ? (
          <button className="btn primary block" onClick={upgrade}>
            Upgrade now
          </button>
        ) : (
          <>
            {upgrading && <div className="muted small">Upgrading…</div>}
            <pre className="upgrade-log">{log.join("")}</pre>
            {done && (
              <div className="muted small">
                Restart the Ollama service to apply.
              </div>
            )}
          </>
        )
      ) : (
        <div className="muted small">
          Ollama is remote — upgrade it on its host.
        </div>
      )}
    </div>
  );
}
