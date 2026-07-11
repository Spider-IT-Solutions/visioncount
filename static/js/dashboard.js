const qs = (id) => document.getElementById(id);
const jsonHeaders = { "Content-Type": "application/json" };

let cfg = null;
let roi = null;
let lines = [];
let dragging = null;
let scrubDragging = false;

function debounce(fn, wait) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), wait);
  };
}

function fetchJSON(url, opts) {
  return fetch(url, opts).then((r) => r.json());
}

function patchConfig(partial) {
  fetch("/api/config", { method: "POST", headers: jsonHeaders, body: JSON.stringify(partial) });
}
const patchConfigDebounced = debounce(patchConfig, 150);

function showMsg(text) {
  qs("action-msg").textContent = text;
  setTimeout(() => (qs("action-msg").textContent = ""), 4000);
}

function setSliderValue(id, val) {
  const el = qs(id);
  el.value = val;
  if (el.nextElementSibling && el.nextElementSibling.tagName === "OUTPUT") {
    el.nextElementSibling.textContent = val;
  }
}

function populateFromConfig(c) {
  cfg = c;
  ["l_h", "l_s", "l_v", "u_h", "u_s", "u_v"].forEach((k) => setSliderValue(k, c.color_calibration.hsv[k]));
  setSliderValue("confidence_threshold", c.detection.confidence_threshold);
  setSliderValue("nms_threshold", c.detection.nms_threshold);
  setSliderValue("sensitivity", c.detection.sensitivity);
  ["min_area", "max_area", "min_width", "max_width", "min_height", "max_height", "aspect_min", "aspect_max"].forEach(
    (k) => (qs(k).value = c.filters[k])
  );

  ["brightness", "contrast", "saturation", "exposure", "gain"].forEach((k) => setSliderValue(k, c.camera[k]));
  qs("rotation").value = c.camera.rotation;
  qs("flip").value = c.camera.flip;
  qs("source_type").value = c.camera.source_type;
  qs("source").value = c.camera.source;
  qs("cam_width").value = c.camera.width;
  qs("cam_height").value = c.camera.height;
  qs("cam_fps").value = c.camera.fps;
  qs("cam_loop").checked = c.camera.loop !== false;

  const pp = c.preprocessing;
  qs("pp_gamma_enabled").checked = pp.gamma.enabled;
  setSliderValue("pp_gamma_value", Math.round(pp.gamma.value * 100));
  qs("pp_sharpen_enabled").checked = pp.sharpen.enabled;
  setSliderValue("pp_sharpen_amount", pp.sharpen.amount);
  qs("pp_blur_enabled").checked = pp.blur.enabled;
  qs("pp_blur_kernel").value = pp.blur.kernel;
  qs("pp_gaussian_blur_enabled").checked = pp.gaussian_blur.enabled;
  qs("pp_gaussian_blur_kernel").value = pp.gaussian_blur.kernel;
  qs("pp_gaussian_blur_sigma").value = pp.gaussian_blur.sigma;
  qs("pp_median_blur_enabled").checked = pp.median_blur.enabled;
  qs("pp_median_blur_kernel").value = pp.median_blur.kernel;
  qs("pp_clahe_enabled").checked = pp.clahe.enabled;
  qs("pp_clahe_clip_limit").value = pp.clahe.clip_limit;
  qs("pp_clahe_tile_grid").value = pp.clahe.tile_grid;
  qs("pp_hist_eq_enabled").checked = pp.hist_eq.enabled;
  qs("pp_denoise_enabled").checked = pp.denoise.enabled;
  qs("pp_denoise_strength").value = pp.denoise.strength;
  qs("pp_morphology_enabled").checked = pp.morphology.enabled;
  qs("pp_morphology_operation").value = pp.morphology.operation;
  qs("pp_morphology_kernel").value = pp.morphology.kernel;
  qs("pp_morphology_iterations").value = pp.morphology.iterations;

  qs("max_distance").value = c.tracking.max_distance;
  qs("object_timeout").value = c.tracking.object_timeout;
  qs("iou_threshold").value = c.tracking.iou_threshold;
  qs("tracker_type").value = c.tracking.tracker_type;

  qs("save_snapshots").checked = c.output.save_snapshots;
  qs("save_videos").checked = c.output.save_videos;
  qs("csv_logging").checked = c.output.csv_logging;
  qs("json_logging").checked = c.output.json_logging;
  qs("show_window").checked = c.output.show_window;
  qs("output_folder").value = c.output.output_folder;

  setSliderValue("frame_skip", c.performance.frame_skip);
  setSliderValue("resize_factor", c.performance.resize_factor);
  qs("thread_count").value = c.performance.thread_count;
  qs("perf_gpu").checked = c.performance.gpu;
  qs("pi_optimization").checked = c.performance.pi_optimization;

  const modelSelect = qs("model-select");
  if (modelSelect.options.length) modelSelect.value = c.detection.backend;

  const projectBadge = qs("project-status");
  projectBadge.textContent = "PROJECT: " + (c.project?.name || "—");
}

function setRoiFields() {
  qs("roi_enabled").checked = !!roi.enabled;
  qs("roi_x").value = roi.x;
  qs("roi_y").value = roi.y;
  qs("roi_w").value = roi.w;
  qs("roi_h").value = roi.h;
}

function postRoi() {
  fetch("/api/roi", { method: "POST", headers: jsonHeaders, body: JSON.stringify(roi) })
    .then((r) => r.json())
    .then((d) => {
      roi = d;
      setRoiFields();
    });
}

function updateRoiFromFields() {
  roi = {
    enabled: qs("roi_enabled").checked,
    x: +qs("roi_x").value,
    y: +qs("roi_y").value,
    w: +qs("roi_w").value,
    h: +qs("roi_h").value,
  };
  postRoi();
}

function postLines() {
  fetch("/api/lines", { method: "POST", headers: jsonHeaders, body: JSON.stringify(lines) })
    .then((r) => r.json())
    .then((d) => {
      lines = d;
      renderLinesList();
    });
}

function renderLinesList() {
  const container = qs("lines-list");
  container.innerHTML = "";
  lines.forEach((line, idx) => {
    const row = document.createElement("div");
    row.className = "line-row";
    row.innerHTML = `
      <span>#${line.id}</span>
      <label><input type="checkbox" data-idx="${idx}" class="line-enabled" ${line.enabled ? "checked" : ""}> On</label>
      <input type="text" class="line-fwd" data-idx="${idx}" value="${line.forward_label}" title="forward crossing label">
      <input type="text" class="line-bwd" data-idx="${idx}" value="${line.backward_label}" title="backward crossing label">
      <button class="btn line-delete" data-idx="${idx}">Delete</button>
    `;
    container.appendChild(row);
  });
  container.querySelectorAll(".line-enabled, .line-fwd, .line-bwd").forEach((el) =>
    el.addEventListener("change", onLineFieldChange)
  );
  container.querySelectorAll(".line-delete").forEach((el) => el.addEventListener("click", onLineDelete));
}

function onLineFieldChange(e) {
  const idx = +e.target.dataset.idx;
  if (e.target.classList.contains("line-enabled")) lines[idx].enabled = e.target.checked;
  if (e.target.classList.contains("line-fwd")) lines[idx].forward_label = e.target.value || "IN";
  if (e.target.classList.contains("line-bwd")) lines[idx].backward_label = e.target.value || "OUT";
  postLines();
}

function onLineDelete(e) {
  const idx = +e.target.dataset.idx;
  lines.splice(idx, 1);
  postLines();
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function dist(x1, y1, x2, y2) {
  return Math.hypot(x1 - x2, y1 - y2);
}

function pointToSegmentDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1, dy = y2 - y1;
  const lenSq = dx * dx + dy * dy;
  let t = lenSq ? ((px - x1) * dx + (py - y1) * dy) / lenSq : 0;
  t = clamp(t, 0, 1);
  return dist(px, py, x1 + t * dx, y1 + t * dy);
}

function hitTest(pos) {
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (dist(pos.x, pos.y, l.x1, l.y1) < 10) return { type: "line-point", idx: i, point: 1 };
    if (dist(pos.x, pos.y, l.x2, l.y2) < 10) return { type: "line-point", idx: i, point: 2 };
  }
  if (roi) {
    if (dist(pos.x, pos.y, roi.x + roi.w, roi.y + roi.h) < 12) return { type: "roi-resize" };
    if (pos.x >= roi.x && pos.x <= roi.x + roi.w && pos.y >= roi.y && pos.y <= roi.y + roi.h) {
      return { type: "roi-move", dx: pos.x - roi.x, dy: pos.y - roi.y };
    }
  }
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (pointToSegmentDist(pos.x, pos.y, l.x1, l.y1, l.x2, l.y2) < 8) {
      return { type: "line-move", idx: i, startX: pos.x, startY: pos.y, orig: { ...l } };
    }
  }
  return null;
}

function applyDrag(pos) {
  const CW = 640, CH = 480;
  if (dragging.type === "line-point") {
    const l = lines[dragging.idx];
    l[`x${dragging.point}`] = Math.round(clamp(pos.x, 0, CW));
    l[`y${dragging.point}`] = Math.round(clamp(pos.y, 0, CH));
  } else if (dragging.type === "roi-resize") {
    roi.w = Math.round(clamp(pos.x - roi.x, 10, CW - roi.x));
    roi.h = Math.round(clamp(pos.y - roi.y, 10, CH - roi.y));
    setRoiFields();
  } else if (dragging.type === "roi-move") {
    roi.x = Math.round(clamp(pos.x - dragging.dx, 0, CW - roi.w));
    roi.y = Math.round(clamp(pos.y - dragging.dy, 0, CH - roi.h));
    setRoiFields();
  } else if (dragging.type === "line-move") {
    const ddx = pos.x - dragging.startX, ddy = pos.y - dragging.startY;
    const l = lines[dragging.idx];
    l.x1 = Math.round(dragging.orig.x1 + ddx);
    l.y1 = Math.round(dragging.orig.y1 + ddy);
    l.x2 = Math.round(dragging.orig.x2 + ddx);
    l.y2 = Math.round(dragging.orig.y2 + ddy);
  }
}

function getCanvasPos(e) {
  const canvas = qs("overlay");
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) * canvas.width) / rect.width,
    y: ((e.clientY - rect.top) * canvas.height) / rect.height,
  };
}

function wireCanvas() {
  const canvas = qs("overlay");
  canvas.addEventListener("mousedown", (e) => {
    dragging = hitTest(getCanvasPos(e));
  });
  canvas.addEventListener("mousemove", (e) => {
    if (dragging) applyDrag(getCanvasPos(e));
  });
  window.addEventListener("mouseup", () => {
    if (dragging) {
      if (dragging.type === "roi-resize" || dragging.type === "roi-move") postRoi();
      if (dragging.type === "line-point" || dragging.type === "line-move") postLines();
    }
    dragging = null;
  });
}

function draw() {
  const canvas = qs("overlay");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (roi) {
    ctx.strokeStyle = roi.enabled ? "#ffffff" : "#555555";
    ctx.lineWidth = 2;
    ctx.strokeRect(roi.x, roi.y, roi.w, roi.h);
    ctx.fillStyle = ctx.strokeStyle;
    ctx.fillRect(roi.x + roi.w - 6, roi.y + roi.h - 6, 12, 12);
  }

  lines.forEach((l) => {
    ctx.strokeStyle = l.enabled ? "#00c8ff" : "#555555";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(l.x1, l.y1);
    ctx.lineTo(l.x2, l.y2);
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    [[l.x1, l.y1], [l.x2, l.y2]].forEach(([x, y]) => {
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();
    });
  });

  requestAnimationFrame(draw);
}

function updateStats() {
  fetchJSON("/api/stats")
    .then((s) => {
      qs("stat-total").textContent = s.counts?.total ?? 0;
      qs("stat-in").textContent = s.counts?.total_in ?? 0;
      qs("stat-out").textContent = s.counts?.total_out ?? 0;
      qs("stat-opm").textContent = s.objects_per_minute ?? 0;
      qs("stat-objects").textContent = s.object_count_current ?? 0;
      qs("stat-conf").textContent = s.avg_confidence ?? 0;
      qs("stat-fps").textContent = s.fps ?? 0;
      qs("stat-proc").textContent = (s.processing_ms ?? 0) + " ms";
      qs("stat-cpu").textContent = (s.system?.cpu_percent ?? 0) + "%";
      qs("stat-ram").textContent = (s.system?.ram_percent ?? 0) + "%";
      qs("stat-temp").textContent = s.system?.temp_c != null ? s.system.temp_c + "°C" : "—";

      const camBadge = qs("cam-status");
      if (s.camera_ended) {
        camBadge.textContent = "CAMERA: ENDED";
        camBadge.className = "badge badge-warn";
      } else if (s.camera_connected) {
        camBadge.textContent = "CAMERA: CONNECTED";
        camBadge.className = "badge badge-on";
      } else {
        camBadge.textContent = "CAMERA: DISCONNECTED";
        camBadge.className = "badge badge-alert";
      }
      const recBadge = qs("rec-status");
      recBadge.textContent = "REC: " + (s.recording ? "ON" : "OFF");
      recBadge.className = "badge " + (s.recording ? "badge-on" : "badge-off");

      const isFile = !!s.camera_is_file;
      qs("frame-controls").style.opacity = isFile ? "1" : "0.4";
      ["btn-pause", "btn-play", "btn-prev-frame", "btn-next-frame", "frame-scrub"].forEach(
        (id) => (qs(id).disabled = !isFile)
      );
      if (isFile && s.camera_frame_count) {
        const scrub = qs("frame-scrub");
        scrub.max = s.camera_frame_count - 1;
        if (!scrubDragging) scrub.value = s.camera_frame_index ?? 0;
        qs("frame-label").textContent = `Frame ${s.camera_frame_index ?? 0} / ${s.camera_frame_count - 1}`;
      } else {
        qs("frame-label").textContent = "Frame — / —";
      }
      qs("btn-pause").classList.toggle("btn-accent", isFile && !s.camera_paused);
      qs("btn-play").classList.toggle("btn-accent", isFile && !!s.camera_paused);
    })
    .catch(() => {});
}

function wireFrameControls() {
  qs("btn-pause").addEventListener("click", () => fetch("/api/camera/pause", { method: "POST" }));
  qs("btn-play").addEventListener("click", () => fetch("/api/camera/resume", { method: "POST" }));
  qs("btn-prev-frame").addEventListener("click", () =>
    fetch("/api/camera/step", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ delta: -1 }) })
  );
  qs("btn-next-frame").addEventListener("click", () =>
    fetch("/api/camera/step", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ delta: 1 }) })
  );
  const scrub = qs("frame-scrub");
  scrub.addEventListener("input", (e) => {
    scrubDragging = true;
    qs("frame-label").textContent = `Frame ${e.target.value} / ${e.target.max}`;
  });
  scrub.addEventListener("change", (e) => {
    fetch("/api/camera/seek", {
      method: "POST",
      headers: jsonHeaders,
      body: JSON.stringify({ frame: +e.target.value }),
    }).then(() => (scrubDragging = false));
  });
}

function wireInputs() {
  ["l_h", "l_s", "l_v", "u_h", "u_s", "u_v"].forEach((k) => {
    qs(k).addEventListener("input", (e) => {
      setSliderValue(k, e.target.value);
      patchConfigDebounced({ color_calibration: { hsv: { [k]: +e.target.value } } });
    });
  });

  ["confidence_threshold", "nms_threshold"].forEach((k) => {
    qs(k).addEventListener("input", (e) => {
      setSliderValue(k, e.target.value);
      patchConfigDebounced({ detection: { [k]: +e.target.value } });
    });
  });
  qs("sensitivity").addEventListener("input", (e) => {
    setSliderValue("sensitivity", e.target.value);
    patchConfigDebounced({ detection: { sensitivity: +e.target.value } });
  });

  ["min_area", "max_area", "min_width", "max_width", "min_height", "max_height", "aspect_min", "aspect_max"].forEach(
    (k) => {
      qs(k).addEventListener("change", (e) => {
        patchConfig({ filters: { [k]: parseFloat(e.target.value) } });
      });
    }
  );

  ["brightness", "contrast", "saturation", "exposure", "gain"].forEach((k) => {
    qs(k).addEventListener("input", (e) => {
      setSliderValue(k, e.target.value);
      patchConfigDebounced({ camera: { [k]: +e.target.value } });
    });
  });
  qs("rotation").addEventListener("change", (e) => patchConfig({ camera: { rotation: +e.target.value } }));
  qs("flip").addEventListener("change", (e) => patchConfig({ camera: { flip: e.target.value } }));

  qs("source_type").addEventListener("change", (e) => patchConfig({ camera: { source_type: e.target.value } }));
  qs("source").addEventListener("change", (e) => patchConfig({ camera: { source: e.target.value } }));
  qs("cam_width").addEventListener("change", (e) => patchConfig({ camera: { width: +e.target.value } }));
  qs("cam_height").addEventListener("change", (e) => patchConfig({ camera: { height: +e.target.value } }));
  qs("cam_fps").addEventListener("change", (e) => patchConfig({ camera: { fps: +e.target.value } }));
  qs("cam_loop").addEventListener("change", (e) => patchConfig({ camera: { loop: e.target.checked } }));

  qs("model-select").addEventListener("change", (e) => {
    if (!e.target.selectedOptions[0].disabled) patchConfig({ detection: { backend: e.target.value } });
  });

  ["roi_enabled", "roi_x", "roi_y", "roi_w", "roi_h"].forEach((id) =>
    qs(id).addEventListener("change", updateRoiFromFields)
  );

  qs("btn-add-line").addEventListener("click", () => {
    lines.push({ x1: 200, y1: 100, x2: 200, y2: 380, enabled: true, forward_label: "IN", backward_label: "OUT" });
    postLines();
  });

  wirePreprocessing();
  wireTrackingOutputPerformance();
}

function wirePreprocessing() {
  const bind = (checkboxId, patchFn) =>
    qs(checkboxId).addEventListener("change", (e) => patchFn(e.target.checked));

  bind("pp_gamma_enabled", (v) => patchConfig({ preprocessing: { gamma: { enabled: v } } }));
  qs("pp_gamma_value").addEventListener("input", (e) => {
    setSliderValue("pp_gamma_value", e.target.value);
    patchConfigDebounced({ preprocessing: { gamma: { value: +e.target.value / 100 } } });
  });

  bind("pp_sharpen_enabled", (v) => patchConfig({ preprocessing: { sharpen: { enabled: v } } }));
  qs("pp_sharpen_amount").addEventListener("input", (e) => {
    setSliderValue("pp_sharpen_amount", e.target.value);
    patchConfigDebounced({ preprocessing: { sharpen: { amount: +e.target.value } } });
  });

  bind("pp_blur_enabled", (v) => patchConfig({ preprocessing: { blur: { enabled: v } } }));
  qs("pp_blur_kernel").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { blur: { kernel: +e.target.value } } })
  );

  bind("pp_gaussian_blur_enabled", (v) => patchConfig({ preprocessing: { gaussian_blur: { enabled: v } } }));
  qs("pp_gaussian_blur_kernel").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { gaussian_blur: { kernel: +e.target.value } } })
  );
  qs("pp_gaussian_blur_sigma").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { gaussian_blur: { sigma: +e.target.value } } })
  );

  bind("pp_median_blur_enabled", (v) => patchConfig({ preprocessing: { median_blur: { enabled: v } } }));
  qs("pp_median_blur_kernel").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { median_blur: { kernel: +e.target.value } } })
  );

  bind("pp_clahe_enabled", (v) => patchConfig({ preprocessing: { clahe: { enabled: v } } }));
  qs("pp_clahe_clip_limit").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { clahe: { clip_limit: +e.target.value } } })
  );
  qs("pp_clahe_tile_grid").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { clahe: { tile_grid: +e.target.value } } })
  );

  bind("pp_hist_eq_enabled", (v) => patchConfig({ preprocessing: { hist_eq: { enabled: v } } }));

  bind("pp_denoise_enabled", (v) => patchConfig({ preprocessing: { denoise: { enabled: v } } }));
  qs("pp_denoise_strength").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { denoise: { strength: +e.target.value } } })
  );

  bind("pp_morphology_enabled", (v) => patchConfig({ preprocessing: { morphology: { enabled: v } } }));
  qs("pp_morphology_operation").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { morphology: { operation: e.target.value } } })
  );
  qs("pp_morphology_kernel").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { morphology: { kernel: +e.target.value } } })
  );
  qs("pp_morphology_iterations").addEventListener("change", (e) =>
    patchConfig({ preprocessing: { morphology: { iterations: +e.target.value } } })
  );
}

function wireTrackingOutputPerformance() {
  qs("tracker_type").addEventListener("change", (e) => {
    if (!e.target.selectedOptions[0].disabled) patchConfig({ tracking: { tracker_type: e.target.value } });
  });
  qs("max_distance").addEventListener("change", (e) => patchConfig({ tracking: { max_distance: +e.target.value } }));
  qs("object_timeout").addEventListener("change", (e) =>
    patchConfig({ tracking: { object_timeout: +e.target.value } })
  );

  ["save_snapshots", "save_videos", "csv_logging", "json_logging", "show_window"].forEach((k) => {
    qs(k).addEventListener("change", (e) => patchConfig({ output: { [k]: e.target.checked } }));
  });
  qs("output_folder").addEventListener("change", (e) => patchConfig({ output: { output_folder: e.target.value } }));

  qs("frame_skip").addEventListener("input", (e) => {
    setSliderValue("frame_skip", e.target.value);
    patchConfigDebounced({ performance: { frame_skip: +e.target.value } });
  });
  qs("resize_factor").addEventListener("input", (e) => {
    setSliderValue("resize_factor", e.target.value);
    patchConfigDebounced({ performance: { resize_factor: +e.target.value } });
  });
}

function wireButtons() {
  qs("btn-restart-video").addEventListener("click", () =>
    fetch("/api/camera/restart", { method: "POST" }).then(() => showMsg("Video restarted"))
  );

  qs("btn-reset-counts").addEventListener("click", () =>
    fetch("/api/counts/reset", { method: "POST" }).then(() => showMsg("Session count reset"))
  );

  qs("btn-snapshot").addEventListener("click", () =>
    fetchJSON("/api/snapshot", { method: "POST" }).then((d) => showMsg("Snapshot saved: " + d.filename))
  );
  qs("btn-record-start").addEventListener("click", () =>
    fetchJSON("/api/record/start", { method: "POST" }).then((d) => showMsg("Recording started: " + d.filename))
  );
  qs("btn-record-stop").addEventListener("click", () =>
    fetchJSON("/api/record/stop", { method: "POST" }).then((d) => showMsg("Recording saved: " + d.filename))
  );

  qs("btn-new-project").addEventListener("click", () => {
    const name = qs("new-project-name").value.trim();
    if (!name) return showMsg("Enter a project name first");
    fetch("/api/projects", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ name }) })
      .then((r) => r.json())
      .then((d) => {
        if (d.error) return showMsg("Error: " + d.error);
        qs("new-project-name").value = "";
        populateFromConfig(d.config);
        refreshRoiAndLines();
        loadProjects();
        showMsg("Created project: " + d.name);
      });
  });
  qs("btn-open-project").addEventListener("click", () => {
    const name = qs("project-select").value;
    if (!name) return;
    fetch("/api/projects/open", { method: "POST", headers: jsonHeaders, body: JSON.stringify({ name }) })
      .then((r) => r.json())
      .then((d) => {
        if (d.error) return showMsg("Error: " + d.error);
        populateFromConfig(d.config);
        refreshRoiAndLines();
        showMsg("Opened project: " + d.name);
      });
  });
  qs("btn-build-app").addEventListener("click", () => {
    const btn = qs("btn-build-app");
    const status = qs("build-status");
    const logEl = qs("build-log");
    const originalText = status.textContent;
    btn.disabled = true;
    logEl.hidden = true;
    logEl.textContent = "";
    status.textContent = "Building… (PyInstaller can take 10-60s, longer on first run)";
    fetchJSON("/api/projects/build", { method: "POST" })
      .then((d) => {
        if (d.success) {
          status.textContent = "Built: " + d.exe_path + (d.log_path ? " (full log: " + d.log_path + ")" : "");
          showMsg("Build succeeded: " + d.exe_path);
        } else {
          status.textContent =
            "Build failed" + (d.log_path ? " — full untruncated log saved to: " + d.log_path : "") +
            ". Error output below (select all + copy to share it):";
          logEl.textContent = d.log_tail || d.error || "unknown error — no output captured";
          logEl.hidden = false;
        }
      })
      .catch(() => {
        status.textContent = originalText;
        showMsg("Build request failed — could not reach the server");
      })
      .finally(() => {
        btn.disabled = false;
      });
  });

  qs("btn-save-project").addEventListener("click", () => {
    fetchJSON("/api/projects/save", { method: "POST" }).then((d) => {
      if (d.error) return showMsg("Error: " + d.error);
      showMsg("Saved config + runtime.py for: " + d.name);
      loadProjects();
    });
  });

  qs("btn-export").addEventListener("click", () => {
    window.location = "/api/config/export";
  });
  qs("btn-import").addEventListener("click", () => {
    const file = qs("import-file").files[0];
    if (!file) return showMsg("Choose a JSON file first");
    file.text().then((raw) =>
      fetch("/api/config/import", { method: "POST", headers: jsonHeaders, body: raw })
        .then((r) => r.json())
        .then((c) => {
          populateFromConfig(c);
          refreshRoiAndLines();
          showMsg("Config imported");
        })
    );
  });
  qs("btn-reset").addEventListener("click", () =>
    fetchJSON("/api/config/reset", { method: "POST" }).then((c) => {
      populateFromConfig(c);
      refreshRoiAndLines();
      showMsg("Defaults restored");
    })
  );
}

function refreshRoiAndLines() {
  fetchJSON("/api/roi").then((r) => {
    roi = r;
    setRoiFields();
  });
  fetchJSON("/api/lines").then((l) => {
    lines = l;
    renderLinesList();
  });
}

function loadBackends() {
  return fetchJSON("/api/backends").then((list) => {
    const sel = qs("model-select");
    sel.innerHTML = "";
    list.forEach((b) => {
      const o = document.createElement("option");
      o.value = b.name;
      o.textContent = b.name + (b.available ? "" : " (Phase 3)");
      o.disabled = !b.available;
      sel.appendChild(o);
    });
    if (cfg) sel.value = cfg.detection.backend;
  });
}

function loadProjects() {
  return fetchJSON("/api/projects").then((list) => {
    const sel = qs("project-select");
    sel.innerHTML = "";
    list.forEach((name) => {
      const o = document.createElement("option");
      o.value = name;
      o.textContent = name;
      sel.appendChild(o);
    });
    if (cfg?.project?.name) sel.value = cfg.project.name;
  });
}

async function init() {
  populateFromConfig(await fetchJSON("/api/config"));
  roi = await fetchJSON("/api/roi");
  setRoiFields();
  lines = await fetchJSON("/api/lines");
  renderLinesList();
  await loadBackends();
  await loadProjects();

  wireInputs();
  wireButtons();
  wireCanvas();
  wireFrameControls();

  requestAnimationFrame(draw);
  updateStats();
  setInterval(updateStats, 500);
}

document.addEventListener("DOMContentLoaded", init);
