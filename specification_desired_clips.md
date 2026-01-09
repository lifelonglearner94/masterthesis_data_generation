### 📹 Video Specifications & Data

* **Resolution:** 256 x 256
* **Clip length:** 32 F.
* **Framerate:** 30 FPS
* *(240Hz, but only save every 8th step)*

---

### 🎬 Scenario & Environment

* **Camera:** Static, 45° view of ground
* **Background:** Checkerboard or Grid
* **Objects:**
* 1 dynamic object (cube) + walls
* Color random per clip, high contrast to ground



---

### 💥 Physics & Action

* **Action:** Force impulse between Frame 1 and Frame 2
* **Important:**
*  Save force vector and Ground truth (Friction, Mass)
*  Always different (random) force + direction


* **Friction:** Drastic differences between Phase A and Phase B
* **Mass:** Random in Phase A_1, fixed in Phase B and A_2

---

### 📂 Dataset Phases & Quantity

| Phase | Number of Clips | Mass | Friction | Description |
| --- | --- | --- | --- | --- |
| **A_1** | 15,000 | Random | Normal | Training dataset with random mass |
| **B** | 1,000 | Fixed | Slippery (OOD) | Out-of-Distribution friction |
| **A_2** | 1,000 | Fixed | Normal | Continuation with fixed mass |

**Total Clips: 17,000**

#### Phase Details:
- **Phase A_1 (Job IDs 0-14999):** Random mass from configured range, normal friction
- **Phase B (Job IDs 15000-15999):** Fixed mass (1.0 kg), slippery/OOD friction
- **Phase A_2 (Job IDs 16000-16999):** Fixed mass (1.0 kg), normal friction

---

### Data Generation Tool

https://github.com/google-research/kubric
