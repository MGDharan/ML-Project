// Frontend logic: map, camera, coco-ssd detection, simple motion heuristics, and audio guidance.
let map, userMarker, destMarker;
let video = document.getElementById('video');
let overlay = document.getElementById('overlay');
let overlayCtx = overlay.getContext('2d');
let model = null;
let detecting = false;
let trackers = [];
let lastAnnounce = 0;
const ANNOUNCE_INTERVAL = 1400; // ms

function speak(text) {
  if (!window.speechSynthesis) return;
  const u = new SpeechSynthesisUtterance(text);
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(u);
}

function initMap() {
  map = L.map('map').setView([0,0], 2);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19
  }).addTo(map);

  map.on('click', function(e){
    setDestination(e.latlng);
  });
}

function setDestination(latlng){
  if (destMarker) map.removeLayer(destMarker);
  destMarker = L.marker(latlng).addTo(map).bindPopup('Destination').openPopup();
  document.getElementById('dest-info').innerText = `Destination set: ${latlng.lat.toFixed(5)}, ${latlng.lng.toFixed(5)}`;
  speak('Destination set.');
}

function getLocation(){
  if (!navigator.geolocation) {
    document.getElementById('loc-status').innerText = 'Geolocation not available';
    return;
  }
  navigator.geolocation.getCurrentPosition(p => {
    const latlng = [p.coords.latitude, p.coords.longitude];
    document.getElementById('loc-status').innerText = `Location: ${latlng[0].toFixed(5)}, ${latlng[1].toFixed(5)}`;
    if (userMarker) map.removeLayer(userMarker);
    userMarker = L.marker(latlng).addTo(map).bindPopup('You').openPopup();
    map.setView(latlng, 18);
    speak('Location acquired. Click map to choose destination.');
  }, err => {
    document.getElementById('loc-status').innerText = 'Location permission denied or unavailable';
  });
}

async function startCamera(){
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.srcObject = stream;
    await video.play();
    overlay.width = video.videoWidth || 640;
    overlay.height = video.videoHeight || 480;
    return true;
  } catch (e) {
    console.error('Camera start failed', e);
    speak('Cannot access camera.');
    return false;
  }
}

function distancePixels(a,b){
  const dx = a.x-b.x, dy = a.y-b.y; return Math.sqrt(dx*dx+dy*dy);
}

function nowMs(){ return performance.now(); }

function matchDetections(detections){
  const matched = [];
  const used = new Set();
  const newTrackers = [];
  const timestamp = nowMs();

  for (const d of detections){
    const cx = d.bbox[0] + d.bbox[2]/2;
    const cy = d.bbox[1] + d.bbox[3]/2;
    let best = null; let bestDist = 1e9;
    for (let i=0;i<trackers.length;i++){
      if (used.has(i)) continue;
      const t = trackers[i];
      if (t.label !== d.class) continue;
      const dist = distancePixels({x:cx,y:cy}, t.lastCenter);
      if (dist < bestDist){ best = i; bestDist = dist; }
    }
    if (best !== null && bestDist < 80){
      // update tracker
      const t = trackers[best];
      const dt = Math.max((timestamp - t.lastTime)/1000, 1e-3);
      const speed = bestDist/dt; // px/sec
      t.lastCenter = {x:cx, y:cy};
      t.lastTime = timestamp;
      t.speed = 0.7 * (t.speed||0) + 0.3 * speed;
      t.bbox = d.bbox;
      newTrackers.push(t);
      used.add(best);
    } else {
      // new tracker
      newTrackers.push({ label: d.class, lastCenter:{x:cx,y:cy}, lastTime: timestamp, speed:0, bbox:d.bbox });
    }
  }
  trackers = newTrackers;
  return trackers;
}

function classifyPersonState(pxSpeed){
  // heuristics in pixels/sec from camera frames
  if (pxSpeed < 10) return 'not moving';
  if (pxSpeed < 60) return 'walking';
  return 'running';
}

function announceTrackers(ts){
  const now = nowMs();
  if (now - lastAnnounce < ANNOUNCE_INTERVAL) return;
  lastAnnounce = now;
  if (trackers.length === 0) return;
  // announce the most relevant object in center area first
  trackers.sort((a,b)=>{
    const ca = Math.abs(a.lastCenter.x - overlay.width/2);
    const cb = Math.abs(b.lastCenter.x - overlay.width/2);
    return ca - cb;
  });
  const t = trackers[0];
  const dx = t.lastCenter.x - overlay.width/2;
  const horiz = Math.abs(dx) < overlay.width*0.15 ? 'center' : (dx < 0 ? 'left' : 'right');
  let desc = `${t.label} ${horiz}`;
  const state = (t.label === 'person') ? classifyPersonState(t.speed) : (t.speed < 10 ? 'not moving' : 'moving');
  desc = `${t.label} ${state} on your ${horiz}`;
  // if destination exists, give navigation hint
  if (destMarker && userMarker){
    const hint = computeNavigationHint();
    speak(`${desc}. ${hint}`);
  } else {
    speak(desc);
  }
}

function computeNavigationHint(){
  // compute bearing from user to destination and give simple guidance
  try {
    const u = userMarker.getLatLng();
    const d = destMarker.getLatLng();
    const brg = bearing(u.lat, u.lng, d.lat, d.lng);
    // attempt to get device heading via DeviceOrientationEvent
    const heading = lastHeading;
    if (heading != null){
      let diff = normalizeBearing(brg - heading);
      if (Math.abs(diff) < 20) return `Destination ahead ${distanceToString(u,d)}.`;
      if (diff > 0) return `Turn right ${Math.abs(Math.round(diff))} degrees then go forward ${distanceToString(u,d)}.`;
      return `Turn left ${Math.abs(Math.round(diff))} degrees then go forward ${distanceToString(u,d)}.`;
    } else {
      // no heading available, give coarse compass direction
      const dir = bearingToCardinal(brg);
      return `Destination is ${dir}, approximately ${distanceToString(u,d)}.`;
    }
  } catch(e){ return ''; }
}

function distanceToString(a,b){
  const meters = haversine(a.lat,a.lng,b.lat,b.lng);
  if (meters < 1000) return `${Math.round(meters)} meters`;
  return `${(meters/1000).toFixed(2)} kilometers`;
}

// haversine distance in meters
function haversine(lat1, lon1, lat2, lon2){
  const toRad = x => x * Math.PI/180;
  const R = 6371000;
  const dLat = toRad(lat2-lat1);
  const dLon = toRad(lon2-lon1);
  const a = Math.sin(dLat/2)*Math.sin(dLat/2) + Math.cos(toRad(lat1))*Math.cos(toRad(lat2))*Math.sin(dLon/2)*Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c;
}

function bearing(lat1, lon1, lat2, lon2){
  const toRad = x => x * Math.PI/180;
  const toDeg = x => x * 180/Math.PI;
  const dLon = toRad(lon2-lon1);
  const y = Math.sin(dLon) * Math.cos(toRad(lat2));
  const x = Math.cos(toRad(lat1))*Math.sin(toRad(lat2)) - Math.sin(toRad(lat1))*Math.cos(toRad(lat2))*Math.cos(dLon);
  return (toDeg(Math.atan2(y,x)) + 360) % 360;
}

function normalizeBearing(b){
  let v = ((b + 180) % 360) - 180; if (v < -180) v += 360; return v;
}

function bearingToCardinal(b){
  const dirs = ['north','north-east','east','south-east','south','south-west','west','north-west'];
  const idx = Math.round(((b%360)/45)) % 8;
  return dirs[idx];
}

let lastHeading = null;
window.addEventListener('deviceorientationabsolute', function(e){ if (e.alpha) lastHeading = 360 - e.alpha; });
window.addEventListener('deviceorientation', function(e){ if (e.alpha) lastHeading = 360 - e.alpha; });

async function detectLoop(){
  if (!detecting) return;
  if (!model) return requestAnimationFrame(detectLoop);
  overlayCtx.clearRect(0,0,overlay.width,overlay.height);
  const predictions = await model.detect(video);
  // draw
  for (const p of predictions){
    const [x,y,w,h] = p.bbox;
    overlayCtx.strokeStyle = '#00FF00'; overlayCtx.lineWidth = 2;
    overlayCtx.strokeRect(x,y,w,h);
    overlayCtx.font = '16px Arial'; overlayCtx.fillStyle = '#00FF00';
    overlayCtx.fillText(`${p.class} ${(p.score*100|0)}%`, x+4, y+16);
  }
  matchDetections(predictions);
  announceTrackers();
  requestAnimationFrame(detectLoop);
}

document.getElementById('get-location').addEventListener('click', getLocation);
document.getElementById('start').addEventListener('click', async function(){
  const ok = await startCamera();
  if (!ok) return;
  if (!model) {
    speak('Loading model, please wait');
    model = await cocoSsd.load();
    speak('Model loaded');
  }
  detecting = true;
  document.getElementById('start').disabled = true;
  document.getElementById('stop').disabled = false;
  detectLoop();
});

document.getElementById('stop').addEventListener('click', function(){
  detecting = false;
  document.getElementById('start').disabled = false;
  document.getElementById('stop').disabled = true;
  speak('Detection stopped');
});

// initialization
initMap();
speak('Prototype ready. Click get location then start detection.');
