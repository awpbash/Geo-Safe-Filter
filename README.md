# GenAI Location Privacy

## The Problem — Personally

Have you ever shared a photo from a trip, a hangout, or just a casual day out, and then later wondered what hidden details you might have accidentally exposed?  

With the rise of powerful AI, turning off location services isn’t enough anymore. A single photo can reveal a lot more than you think—a distinctive building, a street sign, even the type of car parked nearby. Put together, that’s often enough for AI (or worse, a person with bad intentions) to figure out exactly where you were.  

This is a real privacy concern, and it’s something we wanted to tackle head-on.  

---

## Our Inspiration

The spark for this project came from a simple but serious thought: *what if we could use AI to fight back against AI?*  

The same technology that can pinpoint your location could also be repurposed to detect and remove those very clues. Our goal was to build a tool that not only highlights these risks but also empowers people to protect themselves.  

One story that really struck us happened here in Singapore. A group of teenagers filmed a casual dance video at an HDB void deck. Unfortunately, a stalker managed to figure out their exact location simply by zooming in on the block number in the background. That was terrifying—and it made us realize how exposed we all are, even when sharing something innocent.  

That incident was a wake-up call, and it pushed us to start working on proactive solutions.

---

## How We Built It

We built a **full-stack mobile app** designed to make privacy protection simple and seamless.  

**Frontend**  
- Built with React Native for Android and iOS.  
- Users just upload a photo, and the app handles the rest behind the scenes.  
- The interface is clean, intuitive, and designed so anyone can use it without technical knowledge.  

**Backend**  
- The backend is where the heavy lifting happens.  
- First, we use **Gemini** to scan the photo for potential location-revealing clues.  
- Then, with **Grounding DINO**, we pinpoint specific objects like street signs, license plates, and landmarks.  
- Once identified, we apply a masking/noise injection algorithm to blur or obscure those sensitive areas.  
- The result? A privacy-safe image that’s ready to be shared without worry.  

---

## How to Run the App

Want to try it out? Here’s how you can get it running on your own machine.  

### Prerequisites
- Node.js (v18+) and npm  
- React Native development environment (Android Studio or Xcode)  
- Python (v3.8+) and pip  
- Git  

### Step 1: Clone the Repository
```bash
git clone https://github.com/awpbash/Geo-Safe-Filter.git
cd Geo-Safe-Filter
```

## Step 2: Set Up the Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --reload
```

### Step 3: Set Up the Frontend
```bash
cd ../frontend
npm install
```

### Step 4: Configure .env file
```bash
API_KEY = {YOUR_API_KEY}
```

### Running app
```bash
# For Android
npx react-native run-android

# For iOS
npx react-native run-ios
```


