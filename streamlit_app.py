"""
How's My Form? - Modern, Intuitive, Sport-Specific Demo (Pinned OpenAI 0.28.0)

1. Dropdown for selecting a sport/exercise (golf, basketball, etc.).
2. Improved overlays with color-coding for angles.
3. Personal trainer–style ChatGPT feedback.
4. Hardcoded OpenAI API key for DEMO ONLY.
5. Local 'trainer_icon.png' for the logo.

Installation:
    pip install streamlit mediapipe opencv-python openai==0.28.0

Run:
    streamlit run streamlit_app.py
"""

import os
import math
import cv2
import mediapipe as mp
import streamlit as st

# Pinned openai==0.28.0 to avoid ChatCompletion error
import openai

###############################
# !!! DEMO HARD-CODED KEY !!! #
###############################
openai.api_key = "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"

# Set Streamlit page config
st.set_page_config(
    page_title="How's My Form?",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load MediaPipe modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Dictionary of sports/exercises and their "ideal" angle ranges (example only)
SPORTS_CONFIG = {
    "Golf Swing": {
        "left_elbow": (70, 150),
        "left_knee": (60, 140)
    },
    "Basketball Shot": {
        "left_elbow": (80, 160),
        "left_knee": (70, 160)
    },
    "Baseball Swing": {
        "left_elbow": (60, 140),
        "left_knee": (60, 140)
    },
    "Squats": {
        "left_knee": (70, 120),
        "left_hip": (60, 120)
    },
    "Deadlifts": {
        "left_knee": (60, 110),
        "left_hip": (50, 100)
    }
}

def calculate_angle(a, b, c):
    """
    Returns the angle at point b (in degrees) given three points: a, b, c.
    Each point is (x, y) in normalized [0..1] coordinates.
    """
    a_x, a_y = a
    b_x, b_y = b
    c_x, c_y = c

    ba = (a_x - b_x, a_y - b_y)
    bc = (c_x - b_x, c_y - b_y)

    dot_prod = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)

    if mag_ba == 0 or mag_bc == 0:
        return None

    cos_angle = dot_prod / (mag_ba * mag_bc)
    cos_angle = max(min(cos_angle, 1.0), -1.0)  # clamp
    angle = math.degrees(math.acos(cos_angle))
    return angle

def get_joint_indices():
    """
    Map joint names to MediaPipe's Pose landmark indices (left side).
    """
    return {
        "left_shoulder": 11,
        "left_elbow": 13,
        "left_wrist": 15,
        "left_hip": 23,
        "left_knee": 25,
        "left_ankle": 27
    }

def measure_joints(frame_rgb, pose, sport_key):
    """
    Process a frame with MediaPipe Pose, measure angles relevant to chosen sport.
    Return dict {joint_name: angle}, plus pose results for overlay.
    """
    results = pose.process(frame_rgb)
    if not results.pose_landmarks:
        return {}, results

    angles = {}
    lm = results.pose_landmarks.landmark
    indices = get_joint_indices()
    ideal_angles = SPORTS_CONFIG.get(sport_key, {})

    # Define how to measure each joint (triplets)
    joint_triplets = {
        "left_elbow": ("left_shoulder", "left_elbow", "left_wrist"),
        "left_knee": ("left_hip", "left_knee", "left_ankle"),
        "left_hip": ("left_shoulder", "left_hip", "left_knee")  # rough approach
    }

    for joint_name, angle_range in ideal_angles.items():
        if joint_name in joint_triplets:
            a_name, b_name, c_name = joint_triplets[joint_name]
            if all(k in indices for k in (a_name, b_name, c_name)):
                a_lm = lm[indices[a_name]]
                b_lm = lm[indices[b_name]]
                c_lm = lm[indices[c_name]]
                angle_val = calculate_angle(
                    (a_lm.x, a_lm.y),
                    (b_lm.x, b_lm.y),
                    (c_lm.x, c_lm.y)
                )
                if angle_val is not None:
                    angles[joint_name] = angle_val

    return angles, results

def draw_overlay(frame_bgr, results, angles, sport_key):
    """
    Draw landmarks + color-coded angle text near each joint.
    Green if in range, red if out of range.
    """
    h, w, _ = frame_bgr.shape
    mp_drawing.draw_landmarks(
        frame_bgr,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
    )

    ideal_angles = SPORTS_CONFIG.get(sport_key, {})
    indices = get_joint_indices()
    lm = results.pose_landmarks.landmark

    for joint_name, angle_val in angles.items():
        if joint_name in ideal_angles:
            min_a, max_a = ideal_angles[joint_name]
            color = (0, 255, 0) if (min_a <= angle_val <= max_a) else (0, 0, 255)
            b_idx = indices.get(joint_name, None)
            if b_idx is None:
                b_idx = indices["left_elbow"]  # fallback
            x_px, y_px = int(lm[b_idx].x * w), int(lm[b_idx].y * h)
            cv2.putText(
                frame_bgr,
                f"{joint_name}:{int(angle_val)}°",
                (x_px, y_px),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )

def analyze_video_with_overlay(video_bytes, sport_key):
    """
    Reads the uploaded video, measures angles for the chosen sport,
    draws an overlay, and returns (annotated_video_path, summary_text).
    """
    temp_video_path = "temp_video.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(video_bytes)

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        return None, "Error: Could not open video."

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video_path = "annotated_output.mp4"
    out = cv2.VideoWriter(out_video_path, fourcc, out_fps, (w, h))

    frame_count = 0
    angle_sums = {}
    angle_counts = {}

    with mp_pose.Pose(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            angles, results = measure_joints(frame_rgb, pose, sport_key)

            # Accumulate angles
            for jn, val in angles.items():
                angle_sums[jn] = angle_sums.get(jn, 0) + val
                angle_counts[jn] = angle_counts.get(jn, 0) + 1

            # Overlay
            annotated_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            if results.pose_landmarks:
                draw_overlay(annotated_bgr, results, angles, sport_key)

            out.write(annotated_bgr)

    cap.release()
    out.release()

    if frame_count == 0:
        return None, "Error: No frames read from the video."

    summary_lines = []
    for jn, total_val in angle_sums.items():
        avg_val = total_val / angle_counts[jn]
        min_a, max_a = SPORTS_CONFIG[sport_key].get(jn, (0,180))
        in_range = (min_a <= avg_val <= max_a)
        status = "Excellent" if in_range else "Needs Improvement"
        summary_lines.append(
            f"{jn.capitalize()} Avg: {avg_val:.1f}° (Ideal: {min_a}-{max_a}) => {status}"
        )

    summary_text = "\n".join(summary_lines)
    return out_video_path, summary_text

def generate_feedback_personal_trainer(analysis_text, sport_key):
    """
    Use ChatGPT to give personal trainer–style feedback, referencing angles & ranges.
    """
    if not openai.api_key:
        return "OpenAI API key not set."

    prompt = f"""
    You are an experienced personal trainer coaching someone on their {sport_key}.
    Here's their average joint angle analysis:

    {analysis_text}

    Provide friendly, motivational feedback on how they can improve, referencing
    these angles and ideal ranges. Keep it concise but specific.
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.8
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error calling OpenAI API: {str(e)}"

def main():
    # Display a local logo image
    # Make sure "trainer_icon.png" is in the same folder as this script.
    try:
        st.image("trainer_icon.png", width=80)
    except:
        st.write("Logo not found. Place 'trainer_icon.png' in the same folder, or replace with a valid URL.")

    st.title("How's My Form?")
    st.subheader("Modern, Intuitive, and Sport-Specific")

    st.write(
        "Upload a video, choose your sport or exercise, and we'll analyze key joint angles. "
        "Then, a friendly AI 'personal trainer' will give you tips!"
    )

    sport_list = list(SPORTS_CONFIG.keys())
    chosen_sport = st.selectbox("Select your sport or exercise", sport_list)

    uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        with st.spinner("Analyzing your video..."):
            video_bytes = uploaded_file.read()
            annotated_path, analysis_summary = analyze_video_with_overlay(video_bytes, chosen_sport)

        if annotated_path is None:
            st.error(analysis_summary)
        else:
            st.success("Analysis Complete!")
            st.write("**Joint Angle Summary:**")
            st.info(analysis_summary)

            st.write("**Annotated Video with Overlays:**")
            st.video(annotated_path)

            with st.spinner("Getting personalized tips from AI..."):
                feedback = generate_feedback_personal_trainer(analysis_summary, chosen_sport)

            if "Error" in feedback:
                st.error(feedback)
            else:
                st.write("**AI Trainer Feedback:**")
                st.success(feedback)
    else:
        st.info("Please select a sport and upload a video to begin.")

if __name__ == "__main__":
    main()