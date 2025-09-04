# RoadSafe — Streamlit app (always-found recipient, faster email, clean layout)
# -----------------------------------------------------------------------------

import io, os, re, math, gzip, time
from datetime import datetime
from email.message import EmailMessage

import streamlit as st
from PIL import Image, ExifTags
import pandas as pd
import requests
import folium
from streamlit_folium import st_folium

# ML
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights
from huggingface_hub import hf_hub_download

# ---- env ----
from dotenv import load_dotenv
load_dotenv()
GMAIL_USER      = os.getenv("GMAIL_USER")
GMAIL_PASSWORD  = os.getenv("GMAIL_PASSWORD")
HF_REPO_ID      = os.getenv("HF_REPO_ID") or "esdk/my-efficientnet-model"
HF_FILENAME     = os.getenv("HF_FILENAME") or "efficientnet_fp16.pt.gz"
HF_TOKEN        = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
DEMO_RECEIVER   = os.getenv("DEMO_RECEIVER")

# ---- page & css ----
st.set_page_config(
    page_title="RoadSafe — Report a Road Issue",
    page_icon="🛣️",
    layout="centered",
    menu_items={}
)
st.markdown("""
<style>
  .center {text-align:center}
  .muted {opacity:.75}
  .section-title {font-size:1.05rem; font-weight:700; margin: 10px 0 4px}
  .result-value {font-size:2.2rem; font-weight:800; margin: 0 0 6px}
  .qual-ok  {color:#22c55e}
  .qual-mid {color:#a16207}
  .qual-bad {color:#ef4444}
  .coords {font-family: ui-monospace, Menlo, Consolas, "Liberation Mono", monospace}
  .folium-map { margin-bottom: 0 !important; }
  div[data-testid="stVerticalBlock"] > div:has(.folium-map) { margin-bottom: 0 !important; }
  .tiny {font-size: .85rem; opacity: .7}

  /* Hide Streamlit “created by / view source / deploy” bubble & toolbar */
  .stDeployButton, .viewerBadge_container__1QSob, .viewerBadge_link__1S137,
  [data-testid="stStatusWidget"], [data-testid="stToolbar"],
  a[aria-label="View source"], a[href*="share.streamlit.io"], a[href*="github.com"] > img {
    display: none !important;
  }
  .stApp > header { display:none !important; }
  footer { visibility:hidden !important; height:0 !important; }

  @media (max-width: 420px) {
    .footer-links a { font-size: 0.85rem; }
    .footer-sep { margin: 0 10px; }
  }
</style>
""", unsafe_allow_html=True)

# ---- labels/defaults ----
MATERIAL_NAMES = ["asphalt", "concrete", "paving_stones", "unpaved", "sett"]
QUALITY_NAMES  = ["excellent", "good", "intermediate", "bad", "very_bad"]
KL_DEFAULT = (3.139003, 101.686855)

# === MODEL ===
class MultiHeadEffB7(nn.Module):
    def __init__(self, n_type=len(MATERIAL_NAMES), n_qual=len(QUALITY_NAMES)):
        super().__init__()
        base = efficientnet_b7(weights=EfficientNet_B7_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        in_f = base.classifier[1].in_features
        self.mat  = nn.Linear(in_f, n_type)
        self.qual = nn.Linear(in_f, n_qual)
    def forward(self, x):
        z = self.features(x)
        return self.mat(z), self.qual(z)

@st.cache_resource(show_spinner=True)
def get_model_session():
    if not HF_REPO_ID or not HF_FILENAME:
        st.error("HF_REPO_ID / HF_FILENAME not set in env."); st.stop()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiHeadEffB7().to(device)

    path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME, token=HF_TOKEN)
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            state = torch.load(io.BytesIO(f.read()), map_location=device)
    else:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {k.replace("_orig_mod.","").replace("module.",""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    @torch.inference_mode()
    def session(pil_img: Image.Image):
        x = tfm(pil_img).unsqueeze(0).to(device)
        t_logits, q_logits = model(x)
        t_probs = F.softmax(t_logits, dim=1).squeeze(0).cpu().tolist()
        q_probs = F.softmax(q_logits, dim=1).squeeze(0).cpu().tolist()
        t_idx = int(torch.tensor(t_probs).argmax())
        q_idx = int(torch.tensor(q_probs).argmax())
        return {
            "surface_type": MATERIAL_NAMES[t_idx],
            "surface_quality": QUALITY_NAMES[q_idx],
            "surface_type_probs": t_probs,
            "surface_quality_probs": q_probs,
        }
    return session

# ---- EXIF (robust) ----
_GPS_TAGS = next((k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"), 34853)
def _rat_to_float(x):
    try:
        if isinstance(x, tuple) and len(x) == 2:
            num, den = x
            den = float(den) if den not in (0, 0.0) else 1.0
            return float(num) / den
        return float(x)
    except Exception:
        try: return float(str(x))
        except Exception: return None
def _dms_to_deg(dms):
    if not (isinstance(dms, (list, tuple)) and len(dms) == 3): return None
    d = _rat_to_float(dms[0]); m = _rat_to_float(dms[1]); s = _rat_to_float(dms[2])
    if None in (d, m, s): return None
    return d + (m/60.0) + (s/3600.0)
def exif_latlon(img_or_bytes):
    try:
        img = Image.open(io.BytesIO(img_or_bytes)) if isinstance(img_or_bytes,(bytes,bytearray)) else img_or_bytes
        data = None
        if hasattr(img, "getexif"):
            try: data = img.getexif()
            except Exception: data = None
        if not data and hasattr(img, "_getexif"):
            try: data = img._getexif()
            except Exception: data = None
        if not data: raise ValueError("no EXIF")
        try: gps_ifd = data.get_ifd(_GPS_TAGS)
        except Exception: gps_ifd = data.get(_GPS_TAGS) if isinstance(data,dict) else None
        if not gps_ifd: raise ValueError("no GPS IFD")
        gps = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        lat_dms, lon_dms = gps.get("GPSLatitude"), gps.get("GPSLongitude")
        if not lat_dms or not lon_dms: raise ValueError("no GPS lat/lon")
        lat = _dms_to_deg(lat_dms); lon = _dms_to_deg(lon_dms)
        if lat is None or lon is None: raise ValueError("bad DMS")
        if str(gps.get("GPSLatitudeRef","N")).upper().startswith("S"): lat=-lat
        if str(gps.get("GPSLongitudeRef","E")).upper().startswith("W"): lon=-lon
        return (float(lat), float(lon))
    except Exception:
        try:
            import piexif
            buf = io.BytesIO()
            (img_or_bytes if isinstance(img_or_bytes,Image.Image) else Image.open(io.BytesIO(img_or_bytes))).save(buf, format="JPEG")
            ex = piexif.load(buf.getvalue()).get("GPS", {})
            lat_dms, lon_dms = ex.get("GPSLatitude"), ex.get("GPSLongitude")
            lat_ref = (ex.get(piexif.GPSIFD.GPSLatitudeRef,b"N") or b"N").decode().upper()
            lon_ref = (ex.get(piexif.GPSIFD.GPSLongitudeRef,b"E") or b"E").decode().upper()
            if lat_dms and lon_dms:
                rr=lambda v:(v[0]/v[1]) if isinstance(v,tuple) and len(v)==2 and v[1] else float(v)
                lat = rr(lat_dms[0]) + rr(lat_dms[1])/60 + rr(lat_dms[2])/3600
                lon = rr(lon_dms[0]) + rr(lon_dms[1])/60 + rr(lon_dms[2])/3600
                if lat_ref.startswith("S"): lat=-lat
                if lon_ref.startswith("W"): lon=-lon
                return (float(lat), float(lon))
        except Exception:
            pass
        return None

def gmaps_link(lat, lon): return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"

# ---- city->email CSV ----
@st.cache_data(show_spinner=False)
def load_city_email_csv():
    for path in ["city_malaysia.csv","/mnt/data/city_malaysia.csv","ward_offices.csv","/mnt/data/city.csv"]:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if {"city","email"}.issubset(df.columns):
                    df["__key"] = df["city"].astype(str).str.strip().str.lower()
                    return df
            except Exception:
                pass
    return None

# ---- reverse geocoding (fast + cached) ----
@st.cache_data(ttl=86400, show_spinner=False)
def reverse_address(lat: float, lon: float):
    lat = float(round(float(lat), 5)); lon = float(round(float(lon), 5))
    try:
        j = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"format":"jsonv2","lat":lat,"lon":lon,"zoom":17,"addressdetails":1},
            headers={"User-Agent":"roadsafe-demo/1.1"},
            timeout=7
        ).json()
        a = j.get("address", {}) if isinstance(j, dict) else {}
        parts = [
            a.get("road") or a.get("pedestrian") or a.get("footway") or a.get("residential"),
            a.get("neighbourhood") or a.get("suburb") or a.get("village"),
            a.get("city") or a.get("town") or a.get("county"),
            a.get("state"),
            a.get("postcode"),
        ]
        line = ", ".join([p for p in parts if p])
        city_key = (a.get("city") or a.get("town") or a.get("county") or "").strip().lower()
        display_city = a.get("city") or a.get("town") or a.get("county") or "Unknown"
        return (line or None), (city_key or None), display_city
    except Exception:
        return None, None, "Unknown"

# ---- recipient resolver (ALWAYS returns something for UI; uses real target for sending)
def resolve_recipient(lat: float, lon: float):
    df = load_city_email_csv()
    addr, city_key, display_city = reverse_address(lat, lon)
    csv_email = None
    if df is not None and city_key:
        hit = df[df["__key"] == city_key]
        if not hit.empty:
            csv_email = str(hit.iloc[0]["email"])

    # What we SHOW in UI (masked):
    fake_display_email = re.sub(r"[^a-z0-9]+","", (display_city or "wardoffice").lower()) + "@wardoffice.com"
    display_email = csv_email or fake_display_email

    # Where we ACTUALLY send:
    actual_email = DEMO_RECEIVER or GMAIL_USER or csv_email or "noreply@example.com"

    return {
        "pretty_address": addr,
        "display_city": display_city,
        "display_email": display_email,  # for masked UI
        "actual_email": actual_email,    # used by SMTP
        "from_csv": bool(csv_email),
    }

# ---- email helpers ----
def mask_email(e: str) -> str:
    if not e or "@" not in e: return "hidden@wardoffice.com"
    local, domain = e.split("@", 1)
    masked_local = (local[:2] + "***") if len(local) > 2 else (local[:1] + "***")
    return f"{masked_local}@{domain}"

def send_email_html(to_email, subject, html_body, text_fallback, image_bytes=None, image_name="image.jpg"):
    if not GMAIL_USER or not GMAIL_PASSWORD:
        return False, "Email credentials not found (set GMAIL_USER / GMAIL_PASSWORD)."

    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(text_fallback, charset="utf-8")
    msg.add_alternative(html_body, subtype="html")

    # Only embed inline (no separate attachment) and compress to speed up SMTP
    if image_bytes:
        subtype = (image_name.split(".")[-1] or "jpeg").lower()
        msg.get_payload()[1].add_related(image_bytes, maintype="image", subtype=subtype, cid="<photo1>")

    import smtplib
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
            s.login(GMAIL_USER, GMAIL_PASSWORD)
            s.send_message(msg)
        return True, "Email sent"
    except Exception as e:
        return False, f"{e}"

# ---- STATE ----
defaults = dict(
    pred=None, img_bytes=None, img_name=None,
    live_point=None, confirmed_point=None,
    map_center=KL_DEFAULT, map_zoom=12,
    location_confirmed=False, report_sent=False,
    allow_submit_anyway=False, exif_prefilled=False,
    pretty_address=None, recipient_display_email=None, actual_recipient=None,
    sending=False,
)
for k,v in defaults.items(): st.session_state.setdefault(k,v)
st.session_state.setdefault("uploader_key", f"uploader-{int(time.time()*1000)}")

def _hard_reset():
    for k in list(defaults.keys()): st.session_state[k] = defaults[k]
    st.session_state["uploader_key"] = f"uploader-{int(time.time()*1000)}"
    time.sleep(0.05)
    st.rerun()

# ---- header ----
st.markdown("""
<div class='center'>
  <h2 style="margin-bottom:0.2rem;">RoadSafe — Report Road Damage</h2>
</div>
""", unsafe_allow_html=True)

# ---- upload → predict ----
uploader = st.file_uploader(
    "Upload a road photo (JPG/PNG)",
    type=["jpg","jpeg","png"],
    key=st.session_state.uploader_key,
    help="A clear road-surface photo works best."
)

# Reset app state if user cleared the file
if st.session_state.get("img_bytes") is not None and uploader is None:
    _hard_reset()

if uploader is not None:
    img = Image.open(uploader).convert("RGB")
    st.session_state.img_bytes = uploader.getvalue()
    st.session_state.img_name = uploader.name
    st.image(img, caption="Input image", use_container_width=True)

    with st.spinner("Analyzing the image…"):
        session = get_model_session()
        st.session_state.pred = session(img)
    st.session_state.report_sent = False

    # EXIF prefill
    gps = exif_latlon(st.session_state.img_bytes)
    if gps:
        st.session_state.live_point = (float(gps[0]), float(gps[1]))
        st.session_state.map_center = st.session_state.live_point
        st.session_state.map_zoom = 16
        st.session_state.exif_prefilled = True

# ---- results ----
if st.session_state.pred:
    res = st.session_state.pred
    type_txt = res["surface_type"].replace("_"," ").title()
    qual_raw = res["surface_quality"]
    qual_txt = qual_raw.replace("_"," ").title()
    qcls = "qual-ok" if qual_raw in ("excellent","good") else ("qual-mid" if qual_raw=="intermediate" else "qual-bad")

    st.markdown("<div class='section-title'>Surface Type</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-value'>{type_txt}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Surface Quality</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result-value {qcls}'>{qual_txt}</div>", unsafe_allow_html=True)

    if qual_raw in {"excellent", "good", "intermediate"} and not st.session_state.allow_submit_anyway:
        st.info("This road seems to be in a decent state. Are you sure you want to report it?")
        st.session_state.allow_submit_anyway = st.toggle("Report anyway", value=st.session_state.allow_submit_anyway)
        if not st.session_state.allow_submit_anyway:
            st.stop()

    st.divider()
    st.subheader("Confirm location of the damage")

    if st.session_state.exif_prefilled and st.session_state.live_point:
        st.success("Location found from your picture (GPS metadata). Please confirm the pin or click elsewhere to adjust.")
    else:
        st.warning("Location not found from your picture — please click on the map to drop a pin.")

    @st.fragment
    def map_and_followups():
        m = folium.Map(
            location=st.session_state.map_center,
            zoom_start=st.session_state.map_zoom,
            control_scale=True,
            tiles="OpenStreetMap"
        )
        if st.session_state.live_point:
            folium.Marker(st.session_state.live_point, tooltip="Selected location").add_to(m)

        out = st_folium(m, width=720, height=420, key="map-main", returned_objects=["last_clicked"])
        if out and out.get("last_clicked"):
            click = out["last_clicked"]
            if click.get("lat") and click.get("lng"):
                st.session_state.live_point = (float(click["lat"]), float(click["lng"]))
                st.session_state.location_confirmed = False
                st.session_state.confirmed_point = None
                st.session_state.exif_prefilled = False
                for k in ("pretty_address","recipient_display_email","actual_recipient","sending"):
                    st.session_state[k] = None if k != "sending" else False

        if st.session_state.live_point:
            lat, lon = st.session_state.live_point
            st.caption(f"Chosen coordinates: <span class='coords'>{lat:.6f}, {lon:.6f}</span>", unsafe_allow_html=True)

        st.button(
            "Confirm location",
            key="confirm-location-btn",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.live_point is None,
            on_click=lambda: st.session_state.update(
                location_confirmed=True,
                confirmed_point=st.session_state.live_point
            )
        )

        if st.session_state.location_confirmed and st.session_state.confirmed_point and not st.session_state.report_sent:
            st.divider()
            st.subheader("Recipient")

            # Resolve recipient ALWAYS (pretend-found for UI)
            if st.session_state.actual_recipient is None:
                lat, lon = st.session_state.confirmed_point
                rec = resolve_recipient(lat, lon)
                st.session_state.pretty_address       = rec["pretty_address"]
                st.session_state.recipient_display_email = rec["display_email"]
                st.session_state.actual_recipient     = rec["actual_email"]

            st.success("Email address of the concerned ward office found.")

            st.divider()
            st.subheader("Send the report")
            st.caption("Email of the ward office — hidden to avoid spam.")
            st.text_input("Recipient", value=mask_email(st.session_state.recipient_display_email), disabled=True)

            # Build preview
            type_txt = res["surface_type"].replace("_"," ").title()
            qual_txt = res["surface_quality"].replace("_"," ").title()
            latc, lonc = st.session_state.confirmed_point
            when = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            location_link = gmaps_link(latc, lonc)
            damage_addr_line = st.session_state.pretty_address or location_link
            core_text = f"""RoadSafe report (demo)
When: {when}
Surface type: {type_txt}
Surface quality: {qual_txt}
Location: {location_link}
Address (damage): {damage_addr_line}
"""

            # One-click send with spinner + lock
            with st.form("send-form", clear_on_submit=False, border=False):
                st.text_area("Preview (read-only)", value=core_text, height=160, disabled=True)
                comment = st.text_area("Additional comments (optional)", placeholder="Add any useful details…")
                submit = st.form_submit_button("Send report", type="primary", use_container_width=True, disabled=st.session_state.sending)

            if submit and not st.session_state.sending:
                st.session_state.sending = True

                # Compress image for faster SMTP
                img_bytes_send = st.session_state.img_bytes
                img_name_send  = st.session_state.img_name or "photo.jpg"
                try:
                    img_pil = Image.open(io.BytesIO(st.session_state.img_bytes)).convert("RGB")
                    img_pil.thumbnail((1280, 1280))
                    buf = io.BytesIO()
                    img_pil.save(buf, format="JPEG", quality=80, optimize=True, progressive=True)
                    img_bytes_send = buf.getvalue()
                    img_name_send  = (img_name_send.rsplit(".",1)[0] + "_compressed.jpg")
                except Exception:
                    pass

                html = f"""
                <div style="font-family:system-ui;">
                  <h2>RoadSafe report (demo)</h2>
                  <p style="color:#555">When: {when}</p>
                  <img src="cid:photo1" style="max-width:640px;border-radius:8px;margin:8px 0 16px"/>
                  <ul style="line-height:1.6">
                    <li><b>Surface type:</b> {type_txt}</li>
                    <li><b>Surface quality:</b> {qual_txt}</li>
                    <li><b>Location:</b> <a href="{location_link}">{location_link}</a></li>
                    <li><b>Address (damage):</b> {damage_addr_line}</li>
                  </ul>
                  {"<p><b>Additional comments:</b><br>"+comment.replace('\\n','<br>')+"</p>" if comment.strip() else ""}
                  <p class="tiny">This message was generated for demo purposes.</p>
                </div>
                """.strip()

                text = core_text + (f"\nAdditional comments:\n{comment}\n" if comment.strip() else "")

                with st.spinner("Sending report…"):
                    ok, info = send_email_html(
                        to_email=st.session_state.actual_recipient,
                        subject="RoadSafe Report (Demo)",
                        html_body=html,
                        text_fallback=text,
                        image_bytes=img_bytes_send,
                        image_name=img_name_send,
                    )

                st.session_state.sending = False
                if ok:
                    st.session_state.report_sent = True
                    st.success("Thanks! Your report was submitted.")
                    time.sleep(1.2)
                    st.toast("This page will reload if you want to submit another report.", icon="✅")
                    time.sleep(1.8)
                    _hard_reset()
                else:
                    st.error(f"Failed to send email: {info}")

    map_and_followups()

# ---- footer ----
st.divider()
st.markdown("""
<style>
  .footer-links { text-align:center; margin-top:8px; }
  .footer-links a { color:#b91c1c; font-weight:600; text-decoration:none; }
  .footer-links a:hover { color:#ef4444; text-decoration:underline; }
  .footer-sep { color:#9ca3af; margin:0 18px; }
</style>
<div class="footer-links">
  <a href="https://www.linkedin.com/in/arina-w/" target="_blank" rel="noopener">Wahab Arina</a>
  <span class="footer-sep">|</span>
  <a href="https://www.linkedin.com/in/eliesdk/" target="_blank" rel="noopener">Sadaka Elie</a>
  <span class="footer-sep">|</span>
  <a href="https://www.linkedin.com/in/marcelloscuderi/" target="_blank" rel="noopener">Scuderi Marcello</a>
</div>
""", unsafe_allow_html=True)
