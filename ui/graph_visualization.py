import json
from typing import Any

import streamlit as st


ENTITY_COLORS = {
    "Person": "#ffb86b",
    "Organization": "#ff7f8e",
    "Technology": "#6fe7dc",
    "Concept": "#c7ff4a",
    "Product": "#d99cff",
    "Project": "#ffd166",
    "Place": "#72a7ff",
    "Event": "#f4a261",
    "Process": "#9bb7ff",
    "Document": "#e9ece8",
    "Device": "#4dd4ac",
    "System": "#74c0fc",
    "Metric": "#f9c74f",
    "Other": "#a9b5ad",
    "Entity": "#a9b5ad",
}


def format_node_label(name: str) -> str:
    """Make model-generated entity names easier to scan."""

    words = name.replace("_", " ").split()
    small_words = {
        "a", "an", "and", "as", "at", "for", "in",
        "of", "on", "or", "the", "to",
    }
    formatted = []

    for index, word in enumerate(words):
        if word.islower():
            if index > 0 and word in small_words:
                formatted.append(word)
            else:
                formatted.append(
                    word[:1].upper() + word[1:]
                )
        else:
            formatted.append(word)

    return " ".join(formatted)


def build_graph_data(
    entities: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Create compact nodes and deduplicated edges for the UI."""

    entity_types = {
        str(entity.get("name", "")).strip().casefold(): str(
            entity.get("type", "Entity")
        )
        for entity in entities
        if str(entity.get("name", "")).strip()
    }

    node_names: set[str] = set()
    edge_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    for fact in facts:
        source = str(fact.get("source", "")).strip()
        predicate = str(
            fact.get("predicate", "RELATED_TO")
        ).strip()
        target = str(fact.get("target", "")).strip()

        if not source or not target:
            continue

        node_names.update((source, target))
        key = (source, predicate, target)
        confidence = float(fact.get("confidence", 0.0))

        if key not in edge_map:
            edge_map[key] = {
                "source": source,
                "target": target,
                "label": predicate,
                "confidence": confidence,
                "count": 1,
            }
        else:
            edge_map[key]["confidence"] = max(
                edge_map[key]["confidence"],
                confidence,
            )
            edge_map[key]["count"] += 1

    nodes = [
        {
            "id": name,
            "label": format_node_label(name),
            "full_label": name,
            "type": entity_types.get(
                name.casefold(),
                "Entity",
            ),
        }
        for name in sorted(node_names, key=str.casefold)
    ]

    return {
        "nodes": nodes,
        "edges": list(edge_map.values()),
    }


def render_graph_visualization(
    entities: list[dict[str, Any]],
    facts: list[dict[str, Any]],
) -> None:
    """Render an interactive, self-contained knowledge graph."""

    graph_data = build_graph_data(entities, facts)

    if not graph_data["edges"]:
        st.info(
            "Ask a question with connected graph facts "
            "to display the visualization."
        )
        return

    serialized_graph = json.dumps(
        graph_data,
        ensure_ascii=True,
    ).replace("<", "\\u003c").replace(
        ">", "\\u003e"
    ).replace("&", "\\u0026")
    serialized_colors = json.dumps(
        ENTITY_COLORS,
        ensure_ascii=True,
    )

    graph_html = r"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        * { box-sizing: border-box; }
        html {
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
        body {
          margin: 0;
          color: #eef4ec;
          background: #0c100d;
          font-family: Aptos, "Segoe UI", sans-serif;
        }
        .shell {
          overflow: hidden;
          border: 1px solid #2a342d;
          border-radius: 8px;
          background: #0c100d;
        }
        .toolbar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 12px;
          min-height: 58px;
          padding: 10px 14px;
          border-bottom: 1px solid #263129;
          background: #131815;
        }
        .title { font-weight: 750; letter-spacing: .01em; }
        .hint { margin-top: 3px; color: #94a099; font-size: 12px; text-wrap: pretty; }
        .controls { display: flex; gap: 6px; }
        button {
          min-width: 40px;
          min-height: 40px;
          padding: 7px 10px;
          color: #dfe8dd;
          border: 1px solid #3b493f;
          border-radius: 4px;
          background: #18201a;
          cursor: pointer;
          transition: border-color .2s ease, color .2s ease, transform .2s ease;
        }
        button:hover { border-color: #c7ff4a; color: #c7ff4a; }
        button:active { transform: scale(.96); }
        button:focus-visible { outline: 2px solid #c7ff4a; outline-offset: 2px; }
        .stage { position: relative; height: 560px; }
        canvas {
          width: 100%;
          height: 100%;
          display: block;
          cursor: grab;
          touch-action: none;
        }
        canvas.dragging { cursor: grabbing; }
        canvas.hovering { cursor: pointer; }
        canvas:focus-visible { outline: 2px solid #c7ff4a; outline-offset: -3px; }
        .details {
          position: absolute;
          left: 14px;
          bottom: 14px;
          max-width: min(520px, calc(100% - 28px));
          padding: 9px 11px;
          color: #cdd7cf;
          border: 1px solid #334037;
          border-radius: 5px;
          background: rgba(12, 16, 13, .94);
          box-shadow: 0 8px 24px rgba(0, 0, 0, .28);
          font-size: 12px;
          pointer-events: none;
          text-wrap: pretty;
        }
        .legend {
          position: absolute;
          top: 12px;
          left: 12px;
          display: flex;
          flex-wrap: wrap;
          gap: 6px 10px;
          max-width: calc(100% - 24px);
          color: #aeb9b1;
          font-size: 11px;
          pointer-events: none;
        }
        .legend-item { display: inline-flex; align-items: center; gap: 5px; }
        .swatch { width: 8px; height: 8px; border-radius: 50%; }
        @media (max-width: 680px) {
          .toolbar { align-items: flex-start; flex-direction: column; }
          .stage { height: 460px; }
          .controls { width: 100%; }
          .controls button { flex: 1; }
        }
      </style>
    </head>
    <body>
      <section class="shell" aria-label="Interactive knowledge graph">
        <div class="toolbar">
          <div>
            <div class="title">Knowledge graph</div>
            <div class="hint">Labels stay separated. Hover for full names, drag to arrange, and scroll to zoom.</div>
          </div>
          <div class="controls" aria-label="Graph controls">
            <button id="zoom-out" type="button" aria-label="Zoom out">-</button>
            <button id="zoom-in" type="button" aria-label="Zoom in">+</button>
            <button id="reset" type="button">Reset</button>
          </div>
        </div>
        <div class="stage">
          <canvas
            id="graph"
            tabindex="0"
            role="application"
            aria-label="Knowledge graph. Use arrow keys to pan, plus and minus to zoom, and Home to reset."
          ></canvas>
          <div class="legend" id="legend" aria-hidden="true"></div>
          <div class="details" id="details" aria-live="polite">
            Hover over or select a node to inspect it.
          </div>
        </div>
      </section>
      <script>
        const graph = __GRAPH_DATA__;
        const colors = __ENTITY_COLORS__;
        const canvas = document.getElementById("graph");
        const ctx = canvas.getContext("2d");
        const details = document.getElementById("details");
        const stage = canvas.parentElement;
        const byId = new Map();
        let width = 0;
        let height = 0;
        let zoom = 1;
        let panX = 0;
        let panY = 0;
        let dragged = null;
        let selected = null;
        let hovered = null;
        let panning = false;
        let moved = false;
        let lastX = 0;
        let lastY = 0;
        let framesRemaining = 0;
        let frameHandle = null;

        function splitLabel(label, maxCharacters = 22) {
          if (label.length <= maxCharacters) return [label];

          const words = label.split(/\s+/).filter(Boolean);
          const lines = [""];

          words.forEach(word => {
            const current = lines[lines.length - 1];
            const candidate = current ? `${current} ${word}` : word;

            if (candidate.length <= maxCharacters || lines.length === 2) {
              lines[lines.length - 1] = candidate;
            } else {
              lines.push(word);
            }
          });

          if (lines[1] && lines[1].length > maxCharacters) {
            lines[1] = `${lines[1].slice(0, maxCharacters - 1).trim()}...`;
          }

          return lines.slice(0, 2).filter(Boolean);
        }

        function initializeNodeMetrics() {
          graph.nodes.forEach(node => {
            node.lines = splitLabel(node.label);
            const longestLine = Math.max(
              ...node.lines.map(line => line.length),
              8,
            );
            node.width = Math.min(188, Math.max(82, longestLine * 7 + 28));
            node.height = node.lines.length > 1 ? 54 : 40;
            node.radius = Math.hypot(node.width / 2, node.height / 2);
          });
        }

        function placeNodes() {
          byId.clear();
          graph.nodes.forEach((node, index) => {
            const angle = index * 2.399963;
            const radius = index === 0 ? 0 : 65 + Math.sqrt(index) * 62;
            node.x = Math.cos(angle) * radius;
            node.y = Math.sin(angle) * radius;
            node.vx = 0;
            node.vy = 0;
            byId.set(node.id, node);
          });
        }

        initializeNodeMetrics();
        placeNodes();

        const edges = graph.edges
          .map(edge => ({
            ...edge,
            a: byId.get(edge.source),
            b: byId.get(edge.target),
          }))
          .filter(edge => edge.a && edge.b);

        const usedTypes = [...new Set(
          graph.nodes.map(node => node.type)
        )].sort();
        const legend = document.getElementById("legend");

        usedTypes.forEach(type => {
          const item = document.createElement("span");
          item.className = "legend-item";
          const swatch = document.createElement("span");
          swatch.className = "swatch";
          swatch.style.background = colors[type] || colors.Entity;
          const label = document.createElement("span");
          label.textContent = type;
          item.append(swatch, label);
          legend.append(item);
        });

        function wake(frameCount = 120) {
          framesRemaining = Math.max(framesRemaining, frameCount);
          if (frameHandle === null) {
            frameHandle = requestAnimationFrame(frame);
          }
        }

        function fitView() {
          if (!graph.nodes.length) return;

          const minX = Math.min(...graph.nodes.map(
            node => node.x - node.width / 2
          ));
          const maxX = Math.max(...graph.nodes.map(
            node => node.x + node.width / 2
          ));
          const minY = Math.min(...graph.nodes.map(
            node => node.y - node.height / 2
          ));
          const maxY = Math.max(...graph.nodes.map(
            node => node.y + node.height / 2
          ));
          const graphWidth = Math.max(maxX - minX, 1);
          const graphHeight = Math.max(maxY - minY, 1);
          const availableWidth = Math.max(width - 72, 1);
          const availableHeight = Math.max(height - 130, 1);

          zoom = Math.min(
            1.15,
            Math.max(
              .3,
              Math.min(
                availableWidth / graphWidth,
                availableHeight / graphHeight,
              ),
            ),
          );
          panX = width / 2 - ((minX + maxX) / 2) * zoom;
          panY = height / 2 - ((minY + maxY) / 2) * zoom;
        }

        function resetView(resetNodes = false) {
          if (resetNodes) {
            placeNodes();
            for (let index = 0; index < 220; index += 1) {
              simulate();
            }
          }
          fitView();
          selected = null;
          hovered = null;
          details.textContent = "Hover over or select a node to inspect it.";
          wake(2);
        }

        function resize() {
          const ratio = window.devicePixelRatio || 1;
          width = stage.clientWidth;
          height = stage.clientHeight;
          canvas.width = Math.round(width * ratio);
          canvas.height = Math.round(height * ratio);
          ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

          if (panX === 0 && panY === 0) resetView(true);
          else wake(20);
        }

        function worldPoint(event) {
          const rect = canvas.getBoundingClientRect();
          const sx = event.clientX - rect.left;
          const sy = event.clientY - rect.top;
          return {
            x: (sx - panX) / zoom,
            y: (sy - panY) / zoom,
            sx,
            sy,
          };
        }

        function nodeAt(point) {
          for (
            let index = graph.nodes.length - 1;
            index >= 0;
            index -= 1
          ) {
            const node = graph.nodes[index];
            if (
              Math.abs(point.x - node.x) <= node.width / 2
              && Math.abs(point.y - node.y) <= node.height / 2
            ) return node;
          }
          return null;
        }

        function simulate() {
          for (let i = 0; i < graph.nodes.length; i += 1) {
            for (let j = i + 1; j < graph.nodes.length; j += 1) {
              const a = graph.nodes[i];
              const b = graph.nodes[j];
              let dx = b.x - a.x;
              let dy = b.y - a.y;
              const distance = Math.max(Math.hypot(dx, dy), 1);
              const minimumDistance = a.radius + b.radius + 30;
              const overlap = Math.max(
                minimumDistance - distance,
                0,
              );
              const force = overlap > 0
                ? .12 + (overlap / minimumDistance) * .5
                : Math.min(2200 / (distance * distance), .42);
              dx /= distance;
              dy /= distance;

              if (a !== dragged) {
                a.vx -= dx * force;
                a.vy -= dy * force;
              }
              if (b !== dragged) {
                b.vx += dx * force;
                b.vy += dy * force;
              }
            }
          }

          edges.forEach(edge => {
            const dx = edge.b.x - edge.a.x;
            const dy = edge.b.y - edge.a.y;
            const distance = Math.max(Math.hypot(dx, dy), 1);
            const desiredDistance = Math.max(
              190,
              edge.a.radius + edge.b.radius + 72,
            );
            const force = (distance - desiredDistance) * .0018;

            if (edge.a !== dragged) {
              edge.a.vx += (dx / distance) * force;
              edge.a.vy += (dy / distance) * force;
            }
            if (edge.b !== dragged) {
              edge.b.vx -= (dx / distance) * force;
              edge.b.vy -= (dy / distance) * force;
            }
          });

          graph.nodes.forEach(node => {
            if (node === dragged) return;
            node.vx += -node.x * .00012;
            node.vy += -node.y * .00012;
            node.vx *= .86;
            node.vy *= .86;
            node.x += node.vx;
            node.y += node.vy;
          });
        }

        function edgeBoundary(node, ux, uy, padding = 4) {
          const horizontal = (
            node.width / 2 + padding
          ) / Math.max(Math.abs(ux), .001);
          const vertical = (
            node.height / 2 + padding
          ) / Math.max(Math.abs(uy), .001);
          return Math.min(horizontal, vertical);
        }

        function drawArrow(edge) {
          const dx = edge.b.x - edge.a.x;
          const dy = edge.b.y - edge.a.y;
          const distance = Math.max(Math.hypot(dx, dy), 1);
          const ux = dx / distance;
          const uy = dy / distance;
          const startOffset = edgeBoundary(edge.a, ux, uy);
          const endOffset = edgeBoundary(edge.b, ux, uy, 8);
          const startX = edge.a.x + ux * startOffset;
          const startY = edge.a.y + uy * startOffset;
          const endX = edge.b.x - ux * endOffset;
          const endY = edge.b.y - uy * endOffset;
          const alpha = .28 + Math.min(
            Math.max(edge.confidence, 0),
            1,
          ) * .5;

          ctx.strokeStyle = `rgba(153, 170, 159, ${alpha})`;
          ctx.lineWidth = (
            1 + Math.min(edge.count - 1, 3) * .3
          ) / zoom;
          ctx.beginPath();
          ctx.moveTo(startX, startY);
          ctx.lineTo(endX, endY);
          ctx.stroke();

          ctx.fillStyle = `rgba(199, 255, 74, ${alpha + .15})`;
          ctx.beginPath();
          ctx.moveTo(endX, endY);
          ctx.lineTo(
            endX - ux * 9 - uy * 5,
            endY - uy * 9 + ux * 5,
          );
          ctx.lineTo(
            endX - ux * 9 + uy * 5,
            endY - uy * 9 - ux * 5,
          );
          ctx.closePath();
          ctx.fill();

          const focusedNode = hovered || selected;
          const showLabel = (
            focusedNode
            && (edge.a === focusedNode || edge.b === focusedNode)
            && zoom > .58
          );

          if (showLabel) {
            const label = edge.count > 1
              ? `${edge.label.replace(/_/g, " ")} x${edge.count}`
              : edge.label.replace(/_/g, " ");
            ctx.font = `${10 / zoom}px ui-sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "alphabetic";
            const labelX = (startX + endX) / 2;
            const labelY = (startY + endY) / 2 - 5 / zoom;
            const labelWidth = ctx.measureText(label).width;
            ctx.fillStyle = "rgba(12, 16, 13, .9)";
            ctx.fillRect(
              labelX - labelWidth / 2 - 4 / zoom,
              labelY - 9 / zoom,
              labelWidth + 8 / zoom,
              13 / zoom,
            );
            ctx.fillStyle = "#94a099";
            ctx.fillText(label, labelX, labelY);
          }
        }

        function roundedRectangle(x, y, boxWidth, boxHeight, radius) {
          const left = x - boxWidth / 2;
          const top = y - boxHeight / 2;
          const right = left + boxWidth;
          const bottom = top + boxHeight;
          const corner = Math.min(
            radius,
            boxWidth / 2,
            boxHeight / 2,
          );

          ctx.beginPath();
          ctx.moveTo(left + corner, top);
          ctx.lineTo(right - corner, top);
          ctx.quadraticCurveTo(right, top, right, top + corner);
          ctx.lineTo(right, bottom - corner);
          ctx.quadraticCurveTo(right, bottom, right - corner, bottom);
          ctx.lineTo(left + corner, bottom);
          ctx.quadraticCurveTo(left, bottom, left, bottom - corner);
          ctx.lineTo(left, top + corner);
          ctx.quadraticCurveTo(left, top, left + corner, top);
          ctx.closePath();
        }

        function drawNode(node) {
          const active = node === selected || node === hovered;
          const color = colors[node.type] || colors.Entity;
          ctx.shadowColor = active ? color : "transparent";
          ctx.shadowBlur = active ? 16 / zoom : 0;
          ctx.fillStyle = color;
          roundedRectangle(
            node.x,
            node.y,
            node.width,
            node.height,
            11,
          );
          ctx.fill();
          ctx.shadowBlur = 0;
          ctx.strokeStyle = active ? "#f5ffe3" : "#0b0f0c";
          ctx.lineWidth = (active ? 2.5 : 2) / zoom;
          ctx.stroke();

          if (zoom > .38) {
            ctx.font = `700 ${active ? 12 : 11}px ui-sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillStyle = "#10140f";
            const lineHeight = active ? 14 : 13;
            const startY = node.y - (
              (node.lines.length - 1) * lineHeight
            ) / 2;

            node.lines.forEach((line, index) => {
              ctx.fillText(
                line,
                node.x,
                startY + index * lineHeight,
              );
            });
          }
        }

        function draw() {
          ctx.clearRect(0, 0, width, height);
          ctx.save();
          ctx.translate(panX, panY);
          ctx.scale(zoom, zoom);
          edges.forEach(drawArrow);
          graph.nodes.forEach(drawNode);
          ctx.restore();
        }

        function frame() {
          if (!dragged && framesRemaining > 0) simulate();
          draw();
          framesRemaining -= 1;

          if (framesRemaining > 0 || dragged || panning) {
            frameHandle = requestAnimationFrame(frame);
          } else {
            frameHandle = null;
          }
        }

        function setZoom(nextZoom) {
          const centerX = width / 2;
          const centerY = height / 2;
          const worldX = (centerX - panX) / zoom;
          const worldY = (centerY - panY) / zoom;
          zoom = Math.min(2.6, Math.max(.3, nextZoom));
          panX = centerX - worldX * zoom;
          panY = centerY - worldY * zoom;
          wake(2);
        }

        function showNodeDetails(node) {
          const connected = edges.filter(
            edge => edge.a === node || edge.b === node
          ).length;
          details.textContent = (
            `${node.full_label || node.label} · ${node.type} · `
            + `${connected} connection(s)`
          );
        }

        canvas.addEventListener("pointerdown", event => {
          canvas.setPointerCapture(event.pointerId);
          const point = worldPoint(event);
          dragged = nodeAt(point);
          panning = !dragged;
          moved = false;
          lastX = point.sx;
          lastY = point.sy;
          canvas.classList.add("dragging");
          wake(30);
        });

        canvas.addEventListener("pointermove", event => {
          const point = worldPoint(event);

          if (!dragged && !panning) {
            hovered = nodeAt(point);
            canvas.classList.toggle("hovering", Boolean(hovered));

            if (hovered) showNodeDetails(hovered);
            else if (selected) showNodeDetails(selected);
            else {
              details.textContent = (
                "Hover over or select a node to inspect it."
              );
            }
            wake(2);
            return;
          }

          if (
            Math.abs(point.sx - lastX)
            + Math.abs(point.sy - lastY) > 2
          ) moved = true;

          if (dragged) {
            dragged.x = point.x;
            dragged.y = point.y;
            dragged.vx = 0;
            dragged.vy = 0;
          } else {
            panX += point.sx - lastX;
            panY += point.sy - lastY;
          }
          lastX = point.sx;
          lastY = point.sy;
          wake(8);
        });

        canvas.addEventListener("pointerup", event => {
          if (canvas.hasPointerCapture(event.pointerId)) {
            canvas.releasePointerCapture(event.pointerId);
          }

          if (dragged && !moved) {
            selected = dragged;
            showNodeDetails(selected);
          } else if (!dragged && !moved) {
            selected = null;
            details.textContent = (
              "Hover over or select a node to inspect it."
            );
          }

          dragged = null;
          panning = false;
          canvas.classList.remove("dragging");
          wake(70);
        });

        canvas.addEventListener("pointerleave", () => {
          hovered = null;
          canvas.classList.remove("hovering");
          if (selected) showNodeDetails(selected);
          else {
            details.textContent = (
              "Hover over or select a node to inspect it."
            );
          }
          wake(2);
        });

        canvas.addEventListener("wheel", event => {
          event.preventDefault();
          const point = worldPoint(event);
          const nextZoom = Math.min(
            2.6,
            Math.max(
              .3,
              zoom * (event.deltaY < 0 ? 1.12 : .89),
            ),
          );
          panX = point.sx - point.x * nextZoom;
          panY = point.sy - point.y * nextZoom;
          zoom = nextZoom;
          wake(2);
        }, { passive: false });

        canvas.addEventListener("keydown", event => {
          const panStep = 24;
          if (event.key === "ArrowLeft") panX += panStep;
          else if (event.key === "ArrowRight") panX -= panStep;
          else if (event.key === "ArrowUp") panY += panStep;
          else if (event.key === "ArrowDown") panY -= panStep;
          else if (event.key === "+" || event.key === "=") {
            setZoom(zoom * 1.15);
          } else if (event.key === "-") {
            setZoom(zoom * .87);
          } else if (event.key === "Home") {
            resetView(true);
          } else return;

          event.preventDefault();
          wake(2);
        });

        document.getElementById("zoom-in").addEventListener(
          "click",
          () => setZoom(zoom * 1.15),
        );
        document.getElementById("zoom-out").addEventListener(
          "click",
          () => setZoom(zoom * .87),
        );
        document.getElementById("reset").addEventListener(
          "click",
          () => resetView(true),
        );

        new ResizeObserver(resize).observe(stage);
        resize();
        wake(2);
      </script>
    </body>
    </html>
    """.replace("__GRAPH_DATA__", serialized_graph).replace(
        "__ENTITY_COLORS__",
        serialized_colors,
    )

    st.iframe(graph_html, height=635)
