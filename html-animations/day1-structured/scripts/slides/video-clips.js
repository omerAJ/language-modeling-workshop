            // ══════════════════════════════════════
            //  Clip video slides
            // ══════════════════════════════════════
            function isVideoClipSlide(slideId) {
  return VIDEO_CLIP_SLIDE_IDS.includes(slideId);
}

function getVideoClipHost(slideId) {
  return document.querySelector('#slide-' + slideId + ' .clip-video-stage');
}

function getVideoClipElement(slideId) {
  const host = getVideoClipHost(slideId);
  return host ? host.querySelector('video') : null;
}

function formatClipTime(seconds) {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00';
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  if (hours > 0) {
    return String(hours).padStart(2, '0') + ':' + String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
  }
  return String(minutes).padStart(2, '0') + ':' + String(secs).padStart(2, '0');
}

function getVideoCandidates(host) {
  const candidates = [];
  if (!host) return candidates;
  if (host.dataset.videoRelative) {
    const rel = host.dataset.videoRelative;
    candidates.push(rel);
    const fileName = rel.split('/').filter(Boolean).pop();
    if (fileName) {
      candidates.push('file://' + ABS_CLIP_FALLBACK_DIR.replace(/\/+$/, '') + '/' + fileName);
    }
  }
  if (host.dataset.videoPath && host.dataset.videoPath.startsWith('/')) {
    candidates.push('file://' + host.dataset.videoPath);
  } else if (host.dataset.videoPath) {
    candidates.push(host.dataset.videoPath);
  }
  return candidates;
}

function setVideoPlaceholder(host, primary, secondary) {
  if (!host) return;
  const placeholder = host.querySelector('.live-demo-placeholder');
  if (!placeholder) return;
  const extra = secondary ? '<p class="tiny">' + secondary + '</p>' : '';
  placeholder.innerHTML = '<p class="small-text">' + primary + '</p>' + extra;
  placeholder.hidden = false;
}

function hideVideoPlaceholder(host) {
  if (!host) return;
  const placeholder = host.querySelector('.live-demo-placeholder');
  if (placeholder) placeholder.hidden = true;
}

function tryPlayVideo(video) {
  if (!video) return;
  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === 'function') {
    playPromise.catch(function() {});
  }
}

function ensureClipTimeline(host, video) {
  if (!host || !video) return;
  if (host._clipTimelineReady) return;

  const overlay = document.createElement('div');
  overlay.className = 'clip-timeline-overlay';
  overlay.innerHTML =
    '<div class="clip-timeline-main">' +
      '<button class="clip-play-toggle" type="button" aria-label="Pause animation" title="Pause animation">❚❚</button>' +
      '<input type="range" class="clip-seek-slider" min="0" max="1000" step="1" value="0" aria-label="Seek animation timeline">' +
    '</div>' +
    '<div class="clip-timeline-times">' +
      '<span class="clip-time-current">00:00</span>' +
      '<span class="clip-time-total">--:--</span>' +
    '</div>';
  host.appendChild(overlay);

  const playToggle = overlay.querySelector('.clip-play-toggle');
  const slider = overlay.querySelector('.clip-seek-slider');
  const curTime = overlay.querySelector('.clip-time-current');
  const totalTime = overlay.querySelector('.clip-time-total');
  let dragging = false;

  const sync = function(forceSliderVisual) {
    const duration = Number.isFinite(video.duration) && video.duration > 0 ? video.duration : 0;
    const current = Number.isFinite(video.currentTime) && video.currentTime > 0 ? video.currentTime : 0;
    const ratio = duration > 0 ? Math.max(0, Math.min(1, current / duration)) : 0;
    if (!dragging && !forceSliderVisual) {
      slider.value = String(Math.round(ratio * 1000));
    }
    const sliderRatio = Math.max(0, Math.min(1, Number(slider.value || 0) / 1000));
    const visualRatio = dragging ? sliderRatio : ratio;
    slider.style.setProperty('--seek-pct', (visualRatio * 100).toFixed(2) + '%');
    curTime.textContent = formatClipTime(dragging && duration > 0 ? sliderRatio * duration : current);
    totalTime.textContent = duration > 0 ? formatClipTime(duration) : '--:--';
    const paused = video.paused || video.ended;
    playToggle.dataset.state = paused ? 'paused' : 'playing';
    playToggle.textContent = paused ? '▶' : '❚❚';
    playToggle.setAttribute('aria-label', paused ? 'Play animation' : 'Pause animation');
    playToggle.title = paused ? 'Play animation' : 'Pause animation';
  };

  const seekFromSlider = function() {
    if (!Number.isFinite(video.duration) || video.duration <= 0) return;
    const ratio = Math.max(0, Math.min(1, Number(slider.value || 0) / 1000));
    video.currentTime = ratio * video.duration;
  };

  slider.addEventListener('pointerdown', function() {
    dragging = true;
    sync(true);
  });

  slider.addEventListener('pointerup', function() {
    seekFromSlider();
    dragging = false;
    sync();
  });

  slider.addEventListener('pointercancel', function() {
    dragging = false;
    sync();
  });

  slider.addEventListener('input', function() {
    dragging = true;
    seekFromSlider();
    sync(true);
  });

  slider.addEventListener('change', function() {
    seekFromSlider();
    dragging = false;
    sync();
  });

  playToggle.addEventListener('click', function() {
    if (video.paused || video.ended) {
      tryPlayVideo(video);
    } else {
      video.pause();
    }
    sync();
  });

  ['loadedmetadata', 'durationchange', 'timeupdate', 'seeked', 'play', 'pause', 'ended'].forEach(function(evt) {
    video.addEventListener(evt, function() {
      sync();
    });
  });

  host._clipTimelineReady = true;
  host._clipTimelineSync = sync;
  sync();
}

function syncClipTimeline(slideId) {
  const host = getVideoClipHost(slideId);
  if (host && typeof host._clipTimelineSync === 'function') {
    host._clipTimelineSync();
  }
}

function ensureVideoClipLoaded(slideId) {
  const host = getVideoClipHost(slideId);
  if (!host) return null;

  const existing = host.querySelector('video');
  if (existing) {
    ensureClipTimeline(host, existing);
    return existing;
  }

  const candidates = getVideoCandidates(host);
  if (!candidates.length) {
    setVideoPlaceholder(host, 'Video source not found.', 'Check clip path in this slide.');
    return null;
  }

  const video = document.createElement('video');
  video.className = 'live-demo-video';
  video.preload = 'auto';
  video.autoplay = true;
  video.muted = true;
  video.loop = false;
  video.playsInline = true;
  video.controls = false;
  video.setAttribute('playsinline', '');
  ensureClipTimeline(host, video);

  video.addEventListener('loadeddata', function() {
    hideVideoPlaceholder(host);
    tryPlayVideo(video);
    syncClipTimeline(slideId);
  });

  let idx = 0;
  const tryNextSource = () => {
    if (idx >= candidates.length) {
      setVideoPlaceholder(host, 'Unable to load clip.', 'Confirm clip file exists and is readable.');
      return;
    }
    video.src = candidates[idx++];
    video.load();
  };

  video.addEventListener('error', tryNextSource);
  tryNextSource();
  host.prepend(video);
  return video;
}

function playVideoClipSlide(slideId) {
  const video = ensureVideoClipLoaded(slideId);
  if (!video) return;
  try {
    video.currentTime = 0;
  } catch (_err) {}
  tryPlayVideo(video);
  syncClipTimeline(slideId);
}

function pauseVideoClipSlide(slideId, reset) {
  const video = getVideoClipElement(slideId);
  if (!video) return;
  video.pause();
  if (reset) {
    try {
      video.currentTime = 0;
    } catch (_err) {}
  }
  syncClipTimeline(slideId);
}

function pauseAllVideoClipSlides(reset, exceptSlideId) {
  VIDEO_CLIP_SLIDE_IDS.forEach(function(id) {
    if (id === exceptSlideId) return;
    pauseVideoClipSlide(id, reset);
  });
}

function toggleVideoClipPlayback(slideId) {
  const video = ensureVideoClipLoaded(slideId);
  if (!video) return;
  if (video.paused) {
    tryPlayVideo(video);
  } else {
    video.pause();
  }
  syncClipTimeline(slideId);
}
