 Chord & Bassline Generator (Python + Tkinter + MIDI)
A generative music tool that creates modal chord progressions and musical basslines, then exports them as MIDI files.
Designed for producers, composers, and sound designers who want fast idea generation rooted in real music theory.
Built with:
Python
Tkinter (GUI)
Mido (MIDI creation)
Music theory logic (modes, borrowed chords, extensions, voice leading)
 Features
 Modal Scales + Music Theory Engine
Supports major/minor + all church modes:
Ionian, Dorian, Phrygian, Lydian, Mixolydian, Aeolian, Locrian
Borrowed chords / modal interchange options
Progression complexity settings:
Simple
Movement-based
Adventurous (includes deceptive cadences, color tones, etc.)
 Chord Generation
Triads, 7ths, and Extended chords
Optional sus chords (sus2/sus4)
Automatically voiced with basic voice leading
Multiple rhythm styles:
Sustained pads
Pulses
Arpeggiated patterns
 Bassline Generator
Root-only basslines
Moving / passing tone basslines
3rd/5th options
Directional motion toward next chord root
Musical, genre-friendly patterns
 Humanization
Randomized timing
Velocity variation
Selectable intensity levels (none → moderate)
 Graphical Interface (Tkinter)
Choose key, mode, chord complexity, bass complexity, rhythm style, etc.
Choose folder + filename
Generates MIDI in one click
Clean, simple layout
CLI Mode
Run with:
python chord_baseline_generator.py --cli
 Installation
Make sure you have Python 3.10+.
Clone the repo:
git clone https://github.com/will825/chord-bassline-generator.git
cd chord-bassline-generator
Install dependencies:
pip install mido
(If you use a virtual environment, activate it first.)
 Running the GUI
Inside the project folder:
python chord_baseline_generator.py
The GUI will open automatically.
 Output
The script generates a MIDI file containing:
Track 1: Voiced chord progression
Track 2: Bassline
You can import it into:
Logic Pro
Ableton
FL Studio
Cubase
Pro Tools
Any DAW that accepts MIDI
