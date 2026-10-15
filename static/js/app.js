/* ============================================================
   Cricket Biomechanics AI — Front-end Interactions
   Premium analytics dashboard interactivity
   ============================================================ */
(function () {
  "use strict";

  /* -------------------------------------------------------
     Sidebar — mobile toggle
     ------------------------------------------------------- */
  var sidebar = document.getElementById("sidebar");
  var toggle = document.getElementById("sidebarToggle");
  var overlay = document.getElementById("sidebarOverlay");

  function openSidebar() {
    if (sidebar) sidebar.classList.add("open");
    if (overlay) overlay.classList.add("open");
  }
  function closeSidebar() {
    if (sidebar) sidebar.classList.remove("open");
    if (overlay) overlay.classList.remove("open");
  }

  if (toggle) toggle.addEventListener("click", function () {
    if (sidebar && sidebar.classList.contains("open")) closeSidebar();
    else openSidebar();
  });
  if (overlay) overlay.addEventListener("click", closeSidebar);

  /* -------------------------------------------------------
     Active sidebar link
     ------------------------------------------------------- */
  var path = window.location.pathname;
  document.querySelectorAll(".sidebar-link").forEach(function (link) {
    var href = link.getAttribute("href");
    if (!href) return;
    // Strip query/hash for comparison
    var clean = href.split("?")[0].split("#")[0];
    if (clean === "/" && path === "/") {
      link.classList.add("active");
    } else if (clean !== "/" && path.indexOf(clean) === 0) {
      link.classList.add("active");
    }
  });

  /* -------------------------------------------------------
     Animate-in on scroll (Intersection Observer)
     ------------------------------------------------------- */
  if ("IntersectionObserver" in window) {
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });

    document.querySelectorAll(".animate-in").forEach(function (el) {
      observer.observe(el);
    });
  }

  /* -------------------------------------------------------
     Animated counters
     ------------------------------------------------------- */
  function animateCounter(el) {
    var target = parseInt(el.getAttribute("data-counter"), 10);
    if (isNaN(target) || target === 0) return;

    var duration = 1200;
    var start = 0;
    var startTime = null;

    function step(ts) {
      if (!startTime) startTime = ts;
      var progress = Math.min((ts - startTime) / duration, 1);
      // ease-out quad
      var ease = 1 - Math.pow(1 - progress, 3);
      el.textContent = Math.floor(ease * target);
      if (progress < 1) requestAnimationFrame(step);
      else el.textContent = target;
    }

    requestAnimationFrame(step);
  }

  if ("IntersectionObserver" in window) {
    var counterObs = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          animateCounter(entry.target);
          counterObs.unobserve(entry.target);
        }
      });
    }, { threshold: 0.5 });

    document.querySelectorAll("[data-counter]").forEach(function (el) {
      counterObs.observe(el);
    });
  }

  /* -------------------------------------------------------
     Image modal (chart zoom)
     ------------------------------------------------------- */
  var modal = document.getElementById("imageModal");
  var modalImg = document.getElementById("imageModalImg");
  var modalClose = document.getElementById("imageModalClose");

  function openModal(src, alt) {
    if (!modal || !modalImg) return;
    modalImg.src = src;
    modalImg.alt = alt || "";
    modal.classList.add("open");
    document.body.style.overflow = "hidden";
  }

  function closeModal() {
    if (!modal) return;
    modal.classList.remove("open");
    modalImg.src = "";
    document.body.style.overflow = "";
  }

  if (modalClose) modalClose.addEventListener("click", closeModal);
  if (modal) modal.addEventListener("click", function (e) {
    if (e.target === modal) closeModal();
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeModal();
  });

  document.querySelectorAll(".chart-zoomable").forEach(function (img) {
    img.addEventListener("click", function () {
      openModal(img.src, img.alt);
    });
  });

  /* -------------------------------------------------------
     UPLOAD PAGE
     ------------------------------------------------------- */
  var dropzone = document.getElementById("dropzone");
  var fileInput = document.getElementById("videoInput");
  var fileCard = document.getElementById("fileCard");
  var shotOptions = document.querySelectorAll(".shot-opt");
  var uploadForm = document.getElementById("uploadForm");
  var progressPanel = document.getElementById("progressPanel");
  var startBtn = document.getElementById("startBtn");

  function formatSize(bytes) {
    if (!bytes) return "0 B";
    var units = ["B", "KB", "MB", "GB"];
    var i = 0;
    var n = bytes;
    while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
    return n.toFixed(i === 0 ? 0 : 1) + " " + units[i];
  }

  function setFileInfo(file) {
    if (!file || !fileCard) return;
    var nameEl = document.getElementById("fcName");
    var metaEl = document.getElementById("fcMeta");
    if (nameEl) nameEl.textContent = file.name;
    if (metaEl) {
      var ext = file.name.split(".").pop().toUpperCase();
      metaEl.textContent = formatSize(file.size) + "  ·  " + ext + "  ·  " +
        (file.type || "video");
    }
    fileCard.classList.remove("hidden");
  }

  // Drag and drop
  if (dropzone && fileInput) {
    ["dragenter", "dragover"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add("dragover");
      });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dropzone.addEventListener(ev, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove("dragover");
      });
    });
    dropzone.addEventListener("drop", function (e) {
      if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        setFileInfo(e.dataTransfer.files[0]);
      }
    });
    dropzone.addEventListener("click", function () { fileInput.click(); });
    fileInput.addEventListener("change", function () {
      if (fileInput.files.length) setFileInfo(fileInput.files[0]);
    });
  }

  // Shot type cards
  shotOptions.forEach(function (opt) {
    var radio = opt.querySelector('input[type="radio"]');
    opt.addEventListener("click", function () {
      shotOptions.forEach(function (o) { o.classList.remove("selected"); });
      opt.classList.add("selected");
      if (radio) radio.checked = true;
    });
  });

  // Progress steps
  var STEPS = [
    { id: "st-upload",  label: "Video Uploaded" },
    { id: "st-pose",    label: "Detecting Player Pose" },
    { id: "st-land",    label: "Extracting Landmarks" },
    { id: "st-angle",   label: "Calculating Joint Angles" },
    { id: "st-phase",   label: "Detecting Batting Phases" },
    { id: "st-feat",    label: "Extracting Biomechanics Features" },
    { id: "st-results", label: "Generating Results" }
  ];

  function setStep(i, state) {
    if (i >= STEPS.length) return;
    var el = document.getElementById(STEPS[i].id);
    if (!el) return;
    el.className = "step-item " + state;
    var dot = el.querySelector(".step-dot");
    if (dot) dot.textContent = state === "done" ? "\u2713" : (state === "error" ? "!" : "");
  }

  function resetSteps() {
    STEPS.forEach(function (s, i) { setStep(i, "pending"); });
    var text = document.getElementById("progressText");
    if (text) text.textContent = "Uploading video\u2026";
    var bar = document.getElementById("progressBar");
    if (bar) {
      bar.classList.remove("indeterminate");
      bar.style.width = "0%";
    }
  }

  var progressTimers = null;

  function startProgress() {
    resetSteps();
    if (progressPanel) progressPanel.classList.remove("hidden");
    var bar = document.getElementById("progressBar");
    var t = document.getElementById("progressText");
    if (bar) bar.classList.add("indeterminate");
    if (t) {
      t.textContent = "Analysis is running on the server\u2026 this can take " +
        "about a minute per clip. This page reloads automatically when the " +
        "results are ready.";
    }
  }

  if (uploadForm) {
    var uploadSubmitted = false;
    uploadForm.addEventListener("submit", function (e) {
      // Robust double-submit guard: once the form is submitted, ignore every
      // further submit/Enter/click so a single clip can never be uploaded
      // twice (backend also dedups by content hash under a lock).
      // Important: do not disable the file input or radio buttons before submit,
      // because disabled fields are omitted from the multipart POST payload.
      if (uploadSubmitted) {
        e.preventDefault();
        e.stopPropagation();
        return false;
      }
      uploadSubmitted = true;
      if (startBtn) startBtn.disabled = true;
      startProgress();
    });
  }

  // Reset UI if redirected back (validation error)
  if (uploadForm && progressPanel) {
    var hadFlash = document.querySelector(".flash");
    if (hadFlash) {
      progressPanel.classList.add("hidden");
      if (startBtn) startBtn.disabled = false;
    }
  }

  /* -------------------------------------------------------
     TABS (results page)
     ------------------------------------------------------- */
  document.querySelectorAll(".tab").forEach(function (tab) {
    tab.addEventListener("click", function () {
      var target = tab.getAttribute("data-target");
      var tabs = tab.parentElement.querySelectorAll(".tab");
      tabs.forEach(function (t) { t.classList.remove("active"); });
      tab.classList.add("active");
      var container = tab.closest(".card");
      if (!container) return;
      container.querySelectorAll(".tab-panel").forEach(function (p) {
        p.classList.toggle("active", p.id === target);
      });
    });
  });

  /* -------------------------------------------------------
     Shot rating gauge marker (positions the needle on the
     banded 0-10 scale using the rating's data-rating value)
     ------------------------------------------------------- */
  document.querySelectorAll(".rating-gauge").forEach(function (el) {
    var r = parseFloat(el.getAttribute("data-rating"));
    var marker = el.querySelector(".rating-marker");
    if (!marker || isNaN(r)) return;
    marker.style.left = Math.max(0, Math.min(100, r * 10)) + "%";
  });

})();
