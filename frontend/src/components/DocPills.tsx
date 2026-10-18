import type { DocInfo } from "../types";

interface Props {
  /** The documents this chat is answering from, ready ones only. */
  docs: DocInfo[];
  /** Open the sidebar's Documents panel. */
  onOpen: () => void;
  /** Stop using one in this chat (it stays in the library). */
  onRemove: (id: string) => void;
}

// Past this many the bar would crowd out the model pills, so the rest fold into
// a "+N" pill whose tooltip names them.
const MAX_SHOWN = 3;

/** One pill per document the chat is using, named, so it's always visible what the
 *  answer is drawing on. Amber, to read as a third axis beside the vision-model and
 *  image-model pills. */
export default function DocPills({ docs, onOpen, onRemove }: Props) {
  const shown = docs.slice(0, MAX_SHOWN);
  const hidden = docs.slice(MAX_SHOWN);

  return (
    <div className="doc-pills">
      {shown.map((d) => (
        <span key={d.id} className="docs-pill" title={`Answering from ${d.name}`}>
          <button type="button" className="docs-pill-name" onClick={onOpen}>
            {d.kind === "pdf" ? "📕" : "📄"} {d.name}
          </button>
          <button
            type="button"
            className="docs-pill-x"
            title="Stop using this document in this chat"
            aria-label={`Stop using ${d.name} in this chat`}
            onClick={() => onRemove(d.id)}
          >
            ✕
          </button>
        </span>
      ))}
      {hidden.length > 0 && (
        <button
          type="button"
          className="docs-pill docs-pill-more"
          title={hidden.map((d) => d.name).join("\n")}
          onClick={onOpen}
        >
          +{hidden.length} more
        </button>
      )}
    </div>
  );
}
