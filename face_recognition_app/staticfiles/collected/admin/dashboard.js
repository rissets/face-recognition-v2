(() => {
  const parseDataset = (elementId) => {
    const node = document.getElementById(elementId);
    if (!node) {
      return null;
    }
    try {
      return JSON.parse(node.textContent);
    } catch (error) {
      console.warn("[dashboard] Failed to parse dataset", error);
      return null;
    }
  };

  const prepareCanvas = (canvas) => {
    if (!canvas) return null;
    const dpr = window.devicePixelRatio || 1;
    const logicalWidth = canvas.parentElement
      ? canvas.parentElement.clientWidth
      : canvas.clientWidth || 360;
    const logicalHeight = canvas.height;
    canvas.width = logicalWidth * dpr;
    canvas.height = logicalHeight * dpr;
    canvas.style.width = `${logicalWidth}px`;
    canvas.style.height = `${logicalHeight}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, logicalWidth, logicalHeight);
    return { ctx, width: logicalWidth, height: logicalHeight };
  };

  const drawAxes = (ctx, width, height, options = {}) => {
    const { padding = 32, labelColor = "#94a3b8" } = options;
    ctx.strokeStyle = "rgba(148, 163, 184, 0.3)";
    ctx.lineWidth = 1;

    ctx.beginPath();
    ctx.moveTo(padding, padding / 2);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding / 2, height - padding);
    ctx.stroke();

    ctx.fillStyle = labelColor;
    ctx.font = "12px var(--unfold-font-family)";
    ctx.textAlign = "right";
    ctx.fillText("0", padding - 8, height - padding + 12);
  };

  const renderLineChart = (canvasId, payload) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !payload || !payload.labels || !payload.labels.length) return;

    const prepared = prepareCanvas(canvas);
    if (!prepared) return;
    const { ctx, width, height } = prepared;
    const padding = 32;
    const labels = payload.labels;
    const dataset = payload.datasets[0] || { data: [] };

    const maxValue = payload.y_max || Math.max(...dataset.data, 1);
    const minValue = Math.min(...dataset.data, 0);
    const valueRange = Math.max(maxValue - minValue, 1);

    drawAxes(ctx, width, height, { padding });

    // horizontal grid lines
    ctx.strokeStyle = "rgba(148, 163, 184, 0.2)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    const gridLines = 4;
    for (let i = 1; i <= gridLines; i += 1) {
      const y = padding + ((height - padding * 1.5) / gridLines) * i;
      ctx.moveTo(padding, y);
      ctx.lineTo(width - padding / 2, y);
    }
    ctx.stroke();

    const points = dataset.data.map((value, index) => {
      const x =
        padding +
        ((width - padding * 1.5) / Math.max(labels.length - 1, 1)) * index;
      const y =
        height -
        padding -
        ((value - minValue) / valueRange) * (height - padding * 1.5);
      return { x, y };
    });

    if (dataset.fill) {
      ctx.beginPath();
      ctx.moveTo(points[0].x, height - padding);
      points.forEach((point) => ctx.lineTo(point.x, point.y));
      ctx.lineTo(points[points.length - 1].x, height - padding);
      ctx.closePath();
      ctx.fillStyle = `${dataset.color}22`;
      ctx.fill();
    }

    ctx.beginPath();
    ctx.lineWidth = 2;
    ctx.strokeStyle = dataset.color || "#2563eb";
    points.forEach((point, index) => {
      if (index === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.stroke();

    ctx.fillStyle = dataset.color || "#2563eb";
    points.forEach((point) => {
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3, 0, Math.PI * 2);
      ctx.fill();
    });

    // X labels
    ctx.fillStyle = "rgba(30, 41, 59, 0.7)";
    ctx.font = "11px var(--unfold-font-family)";
    ctx.textAlign = "center";
    labels.forEach((label, index) => {
      const point = points[index];
      ctx.fillText(label, point.x, height - padding + 16);
    });
  };

  const renderBarChart = (canvasId, payload) => {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !payload || !payload.labels || !payload.labels.length) return;

    const prepared = prepareCanvas(canvas);
    if (!prepared) return;
    const { ctx, width, height } = prepared;
    const padding = 32;
    const labels = payload.labels;
    const dataset = payload.datasets[0] || { data: [] };
    const maxValue = payload.y_max || Math.max(...dataset.data, 1);

    drawAxes(ctx, width, height, { padding });

    const availableWidth = width - padding * 1.5;
    const barWidth = availableWidth / labels.length * 0.6;
    const gap = availableWidth / labels.length - barWidth;

    dataset.data.forEach((value, index) => {
      const x =
        padding +
        index * (barWidth + gap) +
        gap / 2;
      const barHeight =
        ((value / maxValue) * (height - padding * 1.5)) || 0;
      const y = height - padding - barHeight;

      ctx.fillStyle = dataset.color || "#f97316";
      if (ctx.roundRect) {
        ctx.beginPath();
        ctx.roundRect(x, y, barWidth, barHeight, 6);
        ctx.fill();
      } else {
        ctx.fillRect(x, y, barWidth, barHeight);
      }

      ctx.fillStyle = "rgba(30, 41, 59, 0.7)";
      ctx.font = "11px var(--unfold-font-family)";
      ctx.textAlign = "center";
      ctx.fillText(String(value), x + barWidth / 2, y - 6);
    });

    // X labels
    ctx.fillStyle = "rgba(30, 41, 59, 0.7)";
    ctx.font = "11px var(--unfold-font-family)";
    ctx.textAlign = "center";
    labels.forEach((label, index) => {
      const x =
        padding +
        index * (barWidth + gap) +
        barWidth / 2 +
        gap / 2;
      ctx.fillText(label, x, height - padding + 16);
    });
  };

  const datasets = [
    { canvas: "chart-success-rate", data: parseDataset("chart-success-rate-data"), type: "line" },
    { canvas: "chart-failures", data: parseDataset("chart-failures-data"), type: "bar" },
    { canvas: "chart-request-volume", data: parseDataset("chart-request-volume-data"), type: "line" },
    { canvas: "chart-auth-methods", data: parseDataset("chart-auth-methods-data"), type: "bar" },
    { canvas: "chart-device-types", data: parseDataset("chart-device-types-data"), type: "bar" },
    { canvas: "chart-risk-levels", data: parseDataset("chart-risk-levels-data"), type: "bar" },
  ];

  const render = () => {
    datasets.forEach((entry) => {
      if (!entry.data) return;
      if (entry.type === "bar") {
        renderBarChart(entry.canvas, entry.data);
      } else {
        renderLineChart(entry.canvas, entry.data);
      }
    });
  };

  window.addEventListener("resize", () => {
    window.requestAnimationFrame(render);
  });

  render();
})();
