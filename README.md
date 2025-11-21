
#  **AI Emoji Pose Detector – Meme Generator using MediaPipe & OpenCV**

A fun real-time AI project that detects **hand signs** or **facial expressions**, converts them into **emoji reactions**, and generates a **meme-style output** instantly.

This project uses **Google MediaPipe**, **OpenCV**, and Python to analyze the live camera feed and match poses/faces with predefined emoji templates — then creates a meme preview in a separate output window.

---

## 🚀 Features

* 🎯 **Real-time detection** using MediaPipe (Hands / Face Mesh)
* 👋 **Two modes**:

  * *Hand Tracking Mode*
  * *Face Tracking Mode*
* 🤳 Automatically maps detected gestures or expressions to emojis
* 🖼️ Renders a **meme output** in a separate window (not replacing the camera feed)
* ⬆️ Easy to customize: add your own emojis or poses
* ⚡ Lightweight and works smoothly on most machines

---

## 🛠️ Tech Stack

* **Python 3.10+**
* **OpenCV**
* **MediaPipe**
* **NumPy**

---

## 📂 Project Structure

```
/emojis              → Emoji images with transparency
/memes               → Backgrounds for meme output
main.py              → Main application
pose_detector.py     → Hand/Face detection logic
meme_renderer.py     → Meme generation logic
README.md            → You are here
```

---

## ▶️ How It Works

1. When you start the program, you choose:

   * **Hand Tracking**
   * **Face Tracking**

2. The camera opens and analyzes your gestures or expression.

3. If a pose matches a template — the system loads the corresponding emoji.

4. A meme-style image is generated:

   * Emoji centered
   * Background applied
   * Pose name rendered like typical meme text
   * Shown in a *separate output window*

---

## 📸 Example (Concept)

```
LIVE CAMERA  →  POSE DETECTED →  EMOJI + MEME STYLE OUTPUT
```

---

## 🏁 Installation & Running

### 1. Clone the repo

```bash
git clone (https://github.com/Adham106-lab/emoji_pose_detector.git)
cd your-repo
```

### 2. Install requirements

```bash
pip install opencv-python mediapipe numpy
```

### 3. Run the app

```bash
python main.py
```

---

## 🧩 Customization

Want to add new poses?

1. Put the emoji in the `/emojis` folder
2. Add the pose mapping in the code
3. Done — it will appear automatically in the meme window

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome!
Feel free to fork this project and build on top of it.

---

## ⭐ If you like this project

Please star ⭐ the repository — it helps!
