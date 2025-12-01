"""
CLI and GUI tool that builds modal chord progressions with voiced chords and musical bass,
then renders them to a MIDI file using mido. Includes style presets and humanization.
"""

import os
import random
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional, Sequence, Tuple

import mido
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

# Chromatic scale using sharps only (no enharmonic flats for simplicity)
CHROMATIC = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Interval patterns (in semitones) for each mode relative to the root.
MODE_INTERVALS = {
    "ionian": [0, 2, 4, 5, 7, 9, 11],  # major
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "aeolian": [0, 2, 3, 5, 7, 8, 10],  # natural minor
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}

# Friendly aliases so users can type "major"/"minor" or modal names.
MODE_ALIASES = {
    "major": "ionian",
    "minor": "aeolian",
    "ionian": "ionian",
    "dorian": "dorian",
    "phrygian": "phrygian",
    "lydian": "lydian",
    "mixolydian": "mixolydian",
    "aeolian": "aeolian",
    "locrian": "locrian",
}

# Style defaults for a handful of feels. These are gentle nudges, not strict genre rules.
STYLE_PRESETS: Dict[str, Dict[str, object]] = {
    "ambient": {
        "mode": "lydian",
        "chord_complexity": 3,
        "progression_complexity": 1,
        "rhythm_style": 1,
        "bass_style": "sustain",
        "humanization_amount": 1,
        "tempo": 80,
    },
    "pop": {
        "mode": "ionian",
        "chord_complexity": 2,
        "progression_complexity": 1,
        "rhythm_style": 2,
        "bass_style": "pulse",
        "humanization_amount": 1,
        "tempo": 110,
    },
    "hiphop": {
        "mode": "aeolian",
        "chord_complexity": 3,
        "progression_complexity": 2,
        "rhythm_style": 2,
        "bass_style": "movement",
        "humanization_amount": 2,
        "tempo": 85,
    },
    "trip_hop": {
        "mode": "dorian",
        "chord_complexity": 3,
        "progression_complexity": 2,
        "rhythm_style": 2,
        "bass_style": "movement",
        "humanization_amount": 2,
        "tempo": 78,
    },
    "edm": {
        "mode": "mixolydian",
        "chord_complexity": 2,
        "progression_complexity": 2,
        "rhythm_style": 3,
        "bass_style": "thump",
        "humanization_amount": 1,
        "tempo": 124,
    },
    "dnb": {
        "mode": "dorian",
        "chord_complexity": 2,
        "progression_complexity": 2,
        "rhythm_style": 2,
        "bass_style": "more_movement",
        "humanization_amount": 1,
        "tempo": 172,
    },
    "liquid_dnb": {
        "mode": "lydian",
        "chord_complexity": 3,
        "progression_complexity": 2,
        "rhythm_style": 2,
        "bass_style": "more_movement",
        "humanization_amount": 1,
        "tempo": 174,
    },
    "electronic": {
        "mode": "mixolydian",
        "chord_complexity": 2,
        "progression_complexity": 2,
        "rhythm_style": 3,
        "bass_style": "thump",
        "humanization_amount": 1,
        "tempo": 118,
    },
    "experimental": {
        "mode": "phrygian",
        "chord_complexity": 3,
        "progression_complexity": 3,
        "rhythm_style": 3,
        "bass_style": "movement",
        "humanization_amount": 2,
        "tempo": 95,
    },
    "funk": {
        "mode": "mixolydian",
        "chord_complexity": 3,
        "progression_complexity": 2,
        "rhythm_style": 2,
        "bass_style": "thump",
        "humanization_amount": 1,
        "tempo": 108,
    },
}

# Style-specific progression templates by complexity level.
# Degree specs can be ints (natural degrees) or strings like "b3", "#4".
STYLE_PROGRESSIONS: Dict[str, Dict[int, List[List[object]]]] = {
    "pop": {
        1: [
            [1, 5, 6, 4],
            [1, 6, 4, 5],
            [6, 4, 1, 5],
        ],
        2: [
            [1, 5, 6, 3, 4, 1],
            [2, 5, 1, 6, 4, 5],
        ],
        3: [
            [1, 3, 6, 2, 5, 1],
        ],
    },
    "trip_hop": {
        1: [
            [1, 7, 6, 7],
            [1, 4, "b7", 4],
        ],
        2: [
            [1, "b3", 4, "b7"],
            [6, 4, 1, 5],
        ],
        3: [
            [1, "b3", 4, "b2", 1],
        ],
    },
    "ambient": {
        1: [
            [1, 4, 1, 4],
            [1, 2, 4, 2],
        ],
        2: [
            [1, 4, 6, 4],
        ],
        3: [
            [1, 2, 3, 6],
        ],
    },
    "edm": {
        1: [
            [1, 5, 6, 4],
            [1, 6, 2, 5],
        ],
        2: [
            [1, 5, 6, 3, 4, 1],
        ],
        3: [
            [1, 7, 6, 2, 5, 1],
        ],
    },
    "funk": {
        1: [
            [1, 4, 5, 4],
            [2, 5, 1, 1],
        ],
        2: [
            [1, 4, 2, 5],
        ],
        3: [
            [1, "#4", 4, "b7"],
        ],
    },
    "hiphop": {
        1: [
            [1, 7, 6, 7],
            [1, "b3", 4, "b7"],
        ],
        2: [
            [6, 4, 1, 5],
            [1, "b6", "b7", 4],
        ],
        3: [
            [1, "b5", 4, "b7", "b3"],
        ],
    },
    "dnb": {
        1: [
            [1, 5, 6, 4],
            [6, 4, 1, 5],
        ],
        2: [
            [1, 5, 6, 3, 4, 1],
        ],
        3: [
            [1, 7, 6, 2, 5, 1],
        ],
    },
    "liquid_dnb": {
        1: [
            [1, 4, 1, 4],
            [1, 6, 4, 5],
        ],
        2: [
            [6, 4, 1, 5, 2, 5, 1, 6],
        ],
        3: [
            [1, 2, 5, 6, 4, 5, 1, 2],
        ],
    },
    "electronic": {
        1: [
            [1, 5, 6, 4],
            [4, 1, 5, 6],
        ],
        2: [
            [1, 5, 6, 3, 4, 1],
        ],
        3: [
            [1, "b7", 6, 2, 5, 1],
        ],
    },
    "experimental": {
        1: [
            [1, 4, "b7", 4],
            [1, 2, "b5", 1],
        ],
        2: [
            [1, "b3", 4, "b2"],
        ],
        3: [
            [1, "#4", "b6", 5, "b2"],
        ],
    },
}

@dataclass
class Degree:
    """Represents a scale degree with an optional accidental (e.g., bIII)."""

    scale_degree: int  # 1-based
    accidental: int = 0  # -1 for flat, 0 for natural, +1 for sharp


def note_to_midi(note_name: str, octave: int) -> int:
    """Convert a note name and octave number to a MIDI note value."""
    index = CHROMATIC.index(note_name)
    return index + 12 * (octave + 1)  # MIDI octave numbers start at C-1 = 0


def normalize_mode(mode: str) -> str:
    """Map user input to a canonical mode name."""
    key = mode.strip().lower()
    if key not in MODE_ALIASES:
        raise ValueError(
            "Unsupported mode. Try major, minor, dorian, phrygian, lydian, mixolydian, aeolian, locrian."
        )
    return MODE_ALIASES[key]


def get_style_defaults(style: str) -> Dict[str, object]:
    """Return defaults for a style or empty dict."""
    return STYLE_PRESETS.get(style.lower(), {})


def apply_style_defaults(style: str, params: Dict[str, object]) -> Dict[str, object]:
    """Fill missing params with style defaults without overwriting explicit choices."""
    defaults = get_style_defaults(style)
    merged = dict(params)
    for key, val in defaults.items():
        if merged.get(key) in (None, "", "auto"):
            merged[key] = val
    return merged


def get_mode_intervals(mode: str) -> List[int]:
    """Return the semitone pattern for a mode."""
    normalized = normalize_mode(mode)
    return MODE_INTERVALS[normalized]


def build_scale(root_name: str, mode: str) -> Tuple[List[str], int, List[int]]:
    """Build the modal scale (note names) for the given root and mode."""
    normalized_root = root_name.strip().upper()
    if normalized_root not in CHROMATIC:
        raise ValueError(f"Unsupported key '{root_name}'. Use sharps like C, D#, F#.")

    intervals = get_mode_intervals(mode)
    root_index = CHROMATIC.index(normalized_root)
    scale = []
    for step in intervals:
        note_index = (root_index + step) % len(CHROMATIC)
        scale.append(CHROMATIC[note_index])
    return scale, root_index, intervals


def degree_semitone_offset(intervals: Sequence[int], zero_based_degree: int, accidental: int = 0) -> int:
    """Convert a degree index (0-based) plus accidental into a semitone offset from the root."""
    wraps = zero_based_degree // len(intervals)
    scale_index = zero_based_degree % len(intervals)
    return intervals[scale_index] + 12 * wraps + accidental


def degree_to_string(degree: Degree) -> str:
    """Render a degree as text, e.g., 'b3', '5'."""
    prefix = "b" if degree.accidental < 0 else "#" if degree.accidental > 0 else ""
    return f"{prefix}{degree.scale_degree}"


def parse_degree_spec(spec: object) -> Degree:
    """
    Convert a degree spec (int or string like 'b3', '#4') to a Degree.
    """
    if isinstance(spec, int):
        return Degree(spec, 0)
    if isinstance(spec, str):
        s = spec.strip()
        accidental = 0
        while s and s[0] in ("b", "#"):
            if s[0] == "b":
                accidental -= 1
            elif s[0] == "#":
                accidental += 1
            s = s[1:]
        if not s.isdigit():
            raise ValueError(f"Invalid degree spec: {spec}")
        return Degree(int(s), accidental)
    raise ValueError(f"Unsupported degree spec: {spec}")


# -------------------- Progressions -------------------- #
def choose_borrowed_degree(base: Degree, mode: str, use_borrowed: bool) -> Degree:
    """Light modal interchange: sometimes flatten/raise degrees by mode."""
    if not use_borrowed:
        return base
    if random.random() > 0.25:
        return base
    normalized_mode = normalize_mode(mode)
    borrowed_map = {
        "ionian": [Degree(3, -1), Degree(6, -1), Degree(7, -1), Degree(2, 0)],
        "aeolian": [Degree(5, 0), Degree(7, 0), Degree(3, 0)],
        "dorian": [Degree(7, -1), Degree(4, 0)],
        "mixolydian": [Degree(7, -1), Degree(3, 0)],
        "phrygian": [Degree(2, 0), Degree(7, -1)],
        "lydian": [Degree(2, 0), Degree(5, 0)],
        "locrian": [Degree(4, 0), Degree(7, -1)],
    }
    options = borrowed_map.get(normalized_mode, [])
    if not options:
        return base
    return random.choice(options)


def progression_templates_for_mode(mode: str) -> Dict[str, List[List[int]]]:
    """Mode-aware template pools."""
    majorish = {
        "simple": [
            [1, 5, 6, 4],
            [1, 4, 5, 1],
            [6, 4, 1, 5],
        ],
        "cadential": [
            [1, 2, 5, 1],
            [1, 5, 6, 4],
            [1, 6, 2, 5],
        ],
        "loop": [
            [1, 5, 6, 4, 2, 5, 1, 5],
            [4, 1, 5, 6, 4, 5, 1, 1],
        ],
    }
    minorish = {
        "simple": [
            [1, 7, 6, 7],
            [6, 4, 1, 5],
            [1, 4, 7, 3],
        ],
        "cadential": [
            [1, 4, 5, 1],
            [1, 6, 4, 5],
            [1, 5, 6, 7],
        ],
        "loop": [
            [1, 5, 6, 7, 4, 3, 2, 5],
            [6, 7, 1, 4, 2, 5, 1, 7],
        ],
    }
    modal = {
        "simple": [
            [1, 7, 6, 7],
            [1, 4, 1, 4],
            [1, 5, 4, 1],
        ],
        "cadential": [
            [1, 4, 7, 1],
            [1, 5, 7, 1],
            [1, 4, 5, 1],
        ],
        "loop": [
            [1, 7, 6, 7, 4, 3, 2, 1],
            [1, 5, 4, 5, 1, 7, 6, 7],
        ],
    }
    return majorish if mode in {"ionian", "lydian", "mixolydian"} else minorish if mode in {"aeolian", "dorian", "phrygian"} else modal


def generate_progression_degrees(mode: str, bars: int, complexity: int, use_borrowed: bool, style: str = "custom") -> List[Degree]:
    """
    Build a musical, phrase-like progression: start on I/vi (i/VI), cadences, and repetition.
    Tiles phrases (4/8 bars) to requested length. Enforces tonic start/end and tames borrowing.
    """
    normalized_mode = normalize_mode(mode)

    # Style-aware library
    style_lower = style.lower() if style else ""
    if style_lower in STYLE_PROGRESSIONS:
        style_dict = STYLE_PROGRESSIONS[style_lower]
        # Pick closest complexity bucket available
        if complexity in style_dict:
            pool = style_dict[complexity]
        else:
            keys = sorted(style_dict.keys())
            closest = min(keys, key=lambda k: abs(k - complexity))
            pool = style_dict[closest]
        chosen = random.choice(pool)
        phrase_degrees = [parse_degree_spec(x) for x in chosen]
    else:
        # Fallback to generic templates
        templates = progression_templates_for_mode(normalized_mode)
        if complexity == 1:
            pool = templates["simple"] + templates["cadential"]
        elif complexity == 2:
            pool = templates["cadential"] + templates["loop"]
        else:
            pool = templates["loop"] + templates["cadential"]
        chosen = random.choice(pool)
        phrase_degrees = [Degree(d) for d in chosen]

    phrase_length = max(4, len(phrase_degrees))

    # Tile to requested bars
    needed = (bars + phrase_length - 1) // phrase_length
    tiled = (phrase_degrees * needed)[:bars]

    # Force tonic start/end feel
    if tiled:
        tiled[0] = Degree(1, 0)
        if normalized_mode in {"aeolian", "dorian", "phrygian", "locrian"} and random.random() < 0.2:
            tiled[-1] = Degree(6, 0)
        else:
            tiled[-1] = Degree(1, 0)

    # Optional borrowed color, but never on tonic
    result: List[Degree] = []
    for deg in tiled:
        use_borrow = use_borrowed or complexity == 3
        if use_borrow and deg.scale_degree != 1:
            result.append(choose_borrowed_degree(deg, normalized_mode, True))
        else:
            result.append(deg)
    return result


# -------------------- Chords -------------------- #
def build_chord_name(chord_notes: List[int], sus_type: Optional[str]) -> str:
    """Infer a chord symbol from MIDI pitches (root assumed chord_notes[0])."""
    root_note = CHROMATIC[chord_notes[0] % 12]
    intervals = sorted((n - chord_notes[0]) % 12 for n in chord_notes[1:])
    has_major_third = 4 in intervals
    has_minor_third = 3 in intervals
    has_fifth = 7 in intervals
    has_dim_fifth = 6 in intervals
    seventh = 10 if 10 in intervals else 11 if 11 in intervals else None

    if sus_type:
        quality = sus_type
    elif has_major_third and has_fifth:
        quality = "maj"
    elif has_minor_third and has_fifth:
        quality = "min"
    elif has_minor_third and has_dim_fifth:
        quality = "dim"
    else:
        quality = ""

    suffix = ""
    if seventh is not None:
        if quality == "maj" and seventh == 11:
            suffix = "maj7"
        elif quality == "maj" and seventh == 10:
            suffix = "7"
        elif quality == "min" and seventh == 10:
            suffix = "min7"
        elif quality == "dim" and seventh == 10:
            suffix = "m7b5"
        elif quality == "dim" and seventh == 9:
            suffix = "dim7"
        else:
            suffix = "7"
    else:
        if quality in ("maj", "min", "dim"):
            suffix = quality
        else:
            suffix = ""

    ext_labels = []
    has_ninth = 2 in intervals
    has_eleventh = 5 in intervals
    has_thirteenth = 9 in intervals
    if has_ninth and seventh is not None:
        ext_labels.append("9")
    elif has_ninth:
        ext_labels.append("add9")
    if has_eleventh and seventh is not None:
        ext_labels.append("11")
    elif has_eleventh:
        ext_labels.append("add11")
    if has_thirteenth and seventh is not None:
        ext_labels.append("13")
    elif has_thirteenth:
        ext_labels.append("add13")

    return f"{root_note}{suffix}{''.join(ext_labels)}"


def build_chord_tones(
    root_note: str,
    intervals: Sequence[int],
    degree: Degree,
    chord_complexity: int,
    root_octave: int = 4,
) -> Tuple[List[int], str, Optional[str]]:
    """Build base chord tones (unvoiced) with optional extensions/sus; respect chord_complexity."""
    root_midi_base = note_to_midi(root_note, root_octave)
    steps = [0, 2, 4]  # triad
    sus_type: Optional[str] = None
    extra_ext: List[int] = []

    if chord_complexity >= 2:
        steps.append(6)  # 7th
    if chord_complexity >= 3:
        if random.random() < 0.2:
            sus_type = "sus2" if random.random() < 0.5 else "sus4"
            steps[1] = 1 if sus_type == "sus2" else 3
        if random.random() < 0.6:
            extra_ext.append(8)  # 9th
        if random.random() < 0.35:
            extra_ext.append(10)  # 11th
        if random.random() < 0.25:
            extra_ext.append(12)  # 13th
    steps.extend(extra_ext)

    chord_notes: List[int] = []
    for step in steps:
        zero_based = degree.scale_degree - 1 + step
        offset = degree_semitone_offset(intervals, zero_based, degree.accidental)
        chord_notes.append(root_midi_base + offset)
    chord_notes.sort()
    chord_name = build_chord_name(chord_notes, sus_type)
    return chord_notes, chord_name, sus_type


# -------------------- Voicing -------------------- #
def voice_lead_chord(previous_voicing: Optional[List[int]], target_pitches: List[int], preferred_range: Tuple[int, int] = (48, 72)) -> List[int]:
    """Pick a voicing with minimal movement from previous chord."""
    low, high = preferred_range

    def generate_candidates() -> List[List[int]]:
        base = target_pitches[:]
        candidates: List[List[int]] = []
        shifts = [-24, -12, 0, 12, 24]
        for inversion in range(len(base)):
            rotated = base[inversion:] + [p + 12 for p in base[:inversion]]
            for shift in shifts:
                shifted = [p + shift for p in rotated]
                voiced = []
                prev = None
                for note in shifted:
                    n = note
                    while n < low:
                        n += 12
                    while n > high:
                        n -= 12
                    if prev is not None and n <= prev:
                        while n <= prev:
                            n += 12
                    voiced.append(n)
                    prev = n
                if voiced and (max(voiced) - min(voiced)) <= 24:
                    candidates.append(voiced)
        return candidates

    candidates = generate_candidates()
    if not candidates:
        return target_pitches

    if previous_voicing is None:
        mid = (low + high) / 2
        return min(candidates, key=lambda c: abs(sum(c) / len(c) - mid))

    def movement_score(candidate: List[int]) -> float:
        pairs = zip(candidate, previous_voicing[: len(candidate)])
        return sum(abs(a - b) for a, b in pairs)

    return min(candidates, key=movement_score)


# -------------------- Bass generation -------------------- #
def generate_bass_bar_notes(
    intervals: Sequence[int],
    degree: Degree,
    next_degree: Optional[Degree],
    bass_complexity: int,
    bass_style: str,
    root_note: str,
    mode: str,
    bass_octave: int = 2,
) -> Tuple[List[int], List[float]]:
    """
    Generate bass notes and fractional durations (sum to 1.0 bar) for one bar.
    Styles: sustain, pulse, thump, movement, more_movement.
    """
    base_root = note_to_midi(root_note, bass_octave)
    root_offset = degree_semitone_offset(intervals, degree.scale_degree - 1, degree.accidental)
    current_root = base_root + root_offset
    fifth = base_root + degree_semitone_offset(intervals, degree.scale_degree - 1 + 4, degree.accidental)
    third = base_root + degree_semitone_offset(intervals, degree.scale_degree - 1 + 2, degree.accidental)
    seventh = base_root + degree_semitone_offset(intervals, degree.scale_degree - 1 + 6, degree.accidental)

    def approach_next_root() -> int:
        if not next_degree:
            return current_root
        next_root_offset = degree_semitone_offset(intervals, next_degree.scale_degree - 1, next_degree.accidental)
        next_root = base_root + next_root_offset
        direction = 1 if next_root > current_root else -1
        return next_root - direction

    def clamp(n: int) -> int:
        return max(36, min(n, 60))

    style = bass_style.lower()
    if bass_complexity == 1:
        # Very simple root-focused patterns
        if style == "sustain":
            notes = [clamp(current_root)]
            durations = [1.0]
        else:
            notes = [clamp(current_root), clamp(current_root), clamp(current_root), clamp(fifth)]
            durations = [0.25, 0.25, 0.25, 0.25]
        return notes, durations

    if style == "sustain":
        notes = [clamp(current_root), clamp(fifth)]
        durations = [0.5, 0.5]
    elif style == "pulse":
        notes = [clamp(current_root), clamp(current_root), clamp(fifth), clamp(current_root)]
        durations = [0.25, 0.25, 0.25, 0.25]
    elif style == "thump":
        notes = [clamp(current_root), clamp(fifth), clamp(current_root + 12), clamp(current_root)]
        durations = [0.1875, 0.3125, 0.1875, 0.3125]
    elif style == "movement":
        # Diatonic passing toward next root
        if next_degree:
            current_deg = degree.scale_degree
            next_deg = next_degree.scale_degree
            direction = 1 if next_deg >= current_deg else -1
            passing_deg = max(1, min(7, current_deg + direction))
            passing_offset = degree_semitone_offset(intervals, passing_deg - 1, 0)
            passing = base_root + passing_offset
        else:
            passing_deg = min(7, degree.scale_degree + 1)
            passing_offset = degree_semitone_offset(intervals, passing_deg - 1, 0)
            passing = base_root + passing_offset
        notes = [clamp(current_root), clamp(third), clamp(fifth), clamp(passing)]
        durations = [0.25, 0.25, 0.25, 0.25]
    else:  # more_movement
        if next_degree:
            current_deg = degree.scale_degree
            next_deg = next_degree.scale_degree
            direction = 1 if next_deg >= current_deg else -1
            passing_deg = max(1, min(7, current_deg + direction))
            passing_offset = degree_semitone_offset(intervals, passing_deg - 1, 0)
            passing = base_root + passing_offset
        else:
            passing_deg = min(7, degree.scale_degree + 1)
            passing_offset = degree_semitone_offset(intervals, passing_deg - 1, 0)
            passing = base_root + passing_offset
        upper = clamp(current_root + 12)
        notes = [clamp(current_root), clamp(passing), clamp(fifth), upper]
        durations = [0.1875, 0.1875, 0.25, 0.375]
    return notes, durations


# -------------------- Humanization -------------------- #
def humanize_time(base_ticks: int, humanization_amount: int) -> int:
    """Small onset jitter (±ticks). Amount 0=no jitter, 1≈±10, 2≈±20. Caller keeps bar length intact."""
    if humanization_amount == 0:
        return 0
    rng = 10 if humanization_amount == 1 else 20
    return random.randint(-rng, rng)


def humanize_velocity(base_velocity: int, humanization_amount: int) -> int:
    """Small velocity variation. Amount 0=no change, 1≈±5, 2≈±10."""
    if humanization_amount == 0:
        return base_velocity
    delta = 5 if humanization_amount == 1 else 10
    return max(1, min(127, base_velocity + random.randint(-delta, delta)))


# -------------------- MIDI writing helpers -------------------- #
def write_chord_bar(
    track: MidiTrack,
    voicing: List[int],
    ticks_per_beat: int,
    bar_ticks: int,
    rhythm_style: int,
    humanization_amount: int,
) -> None:
    """Emit chord events for one bar, respecting rhythm_style and humanization."""

    def play_block(duration_ticks: int, notes: List[int]) -> None:
        jitter = humanize_time(duration_ticks // 8, humanization_amount)
        start = max(0, min(duration_ticks - 1, jitter + duration_ticks // 32))
        length = max(1, duration_ticks - start)
        vel = humanize_velocity(80, humanization_amount)
        for i, note in enumerate(notes):
            track.append(Message("note_on", note=note, velocity=vel, channel=0, time=0 if i else start))
        for i, note in enumerate(notes):
            track.append(Message("note_off", note=note, velocity=0, channel=0, time=length if i == 0 else 0))

    if rhythm_style == 1:
        play_block(bar_ticks, voicing)
        return
    if rhythm_style == 2:
        half = bar_ticks // 2
        play_block(half, voicing)
        play_block(bar_ticks - half, voicing)
        return

    # Arpeggiated: 8th-note up/down
    pattern = voicing + voicing[-2:0:-1] if len(voicing) > 2 else voicing
    step = bar_ticks // 8
    for idx in range(8):
        note = pattern[idx % len(pattern)]
        jitter = humanize_time(step // 4, humanization_amount)
        start = max(0, min(step - 1, jitter))
        length = max(1, step - start)
        vel = humanize_velocity(70, humanization_amount)
        track.append(Message("note_on", note=note, velocity=vel, channel=0, time=0 if idx == 0 else start))
        track.append(Message("note_off", note=note, velocity=0, channel=0, time=length))


def write_bass_bar(
    track: MidiTrack,
    bass_notes: List[int],
    durations: List[float],
    bar_ticks: int,
    humanization_amount: int,
) -> None:
    """Emit bass events for one bar using fractional durations (sum to 1.0)."""
    if not bass_notes or not durations:
        return
    ticks = [int(bar_ticks * d) for d in durations]
    diff = bar_ticks - sum(ticks)
    if ticks:
        ticks[-1] += diff  # fix rounding so totals match bar

    for idx, (note, dur) in enumerate(zip(bass_notes, ticks)):
        jitter = humanize_time(dur // 8, humanization_amount)
        start = max(0, min(dur - 1, jitter))
        length = max(1, dur - start)
        vel = humanize_velocity(68, humanization_amount)
        track.append(Message("note_on", note=note, velocity=vel, channel=1, time=0 if idx == 0 else start))
        track.append(Message("note_off", note=note, velocity=0, channel=1, time=length))


# -------------------- MIDI creation -------------------- #
def create_midi(
    root_note: str,
    intervals: Sequence[int],
    progression: List[Degree],
    bars: int,
    filepath: str,
    chord_complexity: int,
    bass_complexity: int,
    bass_style: str,
    rhythm_style: int,
    humanization_amount: int,
    mode: str,
    tempo: int,
    chord_octave: int = 4,
    bass_octave: int = 2,
) -> Tuple[List[str], List[str]]:
    """Build and save the MIDI file with chord and bass tracks."""
    midi_file = MidiFile(ticks_per_beat=480)
    chord_track = MidiTrack()
    bass_track = MidiTrack()
    midi_file.tracks.append(chord_track)
    midi_file.tracks.append(bass_track)

    chord_track.append(MetaMessage("set_tempo", tempo=bpm2tempo(tempo), time=0))
    chord_track.append(Message("program_change", program=0, channel=0, time=0))
    bass_track.append(Message("program_change", program=32, channel=1, time=0))

    bar_ticks = 4 * midi_file.ticks_per_beat  # 4/4 time

    degree_labels: List[str] = []
    chord_names: List[str] = []
    previous_voicing: Optional[List[int]] = None

    for idx in range(bars):
        degree = progression[idx]
        next_degree = progression[idx + 1] if idx + 1 < bars else None

        chord_base, chord_name, _ = build_chord_tones(root_note, intervals, degree, chord_complexity, root_octave=chord_octave)
        voicing = voice_lead_chord(previous_voicing, chord_base, preferred_range=(52, 76))
        previous_voicing = voicing

        bass_notes, bass_durations = generate_bass_bar_notes(
            intervals, degree, next_degree, bass_complexity, bass_style, root_note=root_note, mode=mode, bass_octave=bass_octave
        )

        degree_labels.append(degree_to_string(degree))
        chord_names.append(chord_name)

        write_chord_bar(chord_track, voicing, midi_file.ticks_per_beat, bar_ticks, rhythm_style, humanization_amount)
        write_bass_bar(bass_track, bass_notes, bass_durations, bar_ticks, humanization_amount)

    midi_file.save(filepath)
    return degree_labels, chord_names


# -------------------- CLI helpers -------------------- #
def prompt_int(prompt: str, default: int, min_value: int = 1) -> int:
    raw = input(prompt).strip()
    if not raw:
        return default
    value = int(raw)
    if value < min_value:
        raise ValueError(f"Value must be at least {min_value}.")
    return value


def prompt_yes_no(prompt: str) -> bool:
    raw = input(prompt).strip().lower()
    return raw.startswith("y")


def prompt_with_default(prompt: str, default: str) -> str:
    raw = input(f"{prompt} (default {default}): ").strip()
    return raw or default


# -------------------- Core generation entry -------------------- #
def generate_midi(
    key_input: str,
    mode_input: str,
    bars: int,
    chord_complexity: int,
    progression_complexity: int,
    bass_complexity: int,
    bass_style: str,
    rhythm_style: int,
    humanization_amount: int,
    use_borrowed: bool,
    filename: str,
    folder: str,
    style: Optional[str] = None,
    tempo_override: Optional[int] = None,
) -> Tuple[List[str], List[str], str]:
    """Shared entry used by CLI and GUI."""
    target_folder = folder or os.getcwd()
    if not os.path.isdir(target_folder):
        print(f"Warning: folder '{target_folder}' does not exist. Using current directory instead.")
        target_folder = os.getcwd()

    params = {
        "mode": mode_input,
        "chord_complexity": chord_complexity,
        "progression_complexity": progression_complexity,
        "bass_complexity": bass_complexity,
        "bass_style": bass_style,
        "rhythm_style": rhythm_style,
        "humanization_amount": humanization_amount,
        "tempo": tempo_override,
    }
    params = apply_style_defaults(style or "custom", params)

    normalized_mode = normalize_mode(str(params["mode"]))
    chord_complexity = int(params["chord_complexity"])
    progression_complexity = int(params["progression_complexity"])
    bass_complexity = int(params["bass_complexity"])
    bass_style = str(params["bass_style"])
    rhythm_style = int(params["rhythm_style"])
    humanization_amount = int(params["humanization_amount"])
    tempo = int(params.get("tempo") or 100)

    scale_notes, _, intervals = build_scale(key_input, normalized_mode)
    progression = generate_progression_degrees(normalized_mode, bars, progression_complexity, use_borrowed, style=style or "custom")
    output_path = os.path.abspath(os.path.join(target_folder, filename))

    degree_labels, chord_names = create_midi(
        root_note=scale_notes[0],
        intervals=intervals,
        progression=progression,
        bars=bars,
        filepath=output_path,
        chord_complexity=chord_complexity,
        bass_complexity=bass_complexity,
        bass_style=bass_style,
        rhythm_style=rhythm_style,
        humanization_amount=humanization_amount,
        mode=normalized_mode,
        tempo=tempo,
    )

    return degree_labels, chord_names, output_path


# -------------------- CLI -------------------- #
def main() -> None:
    """Handle user input and orchestrate MIDI creation via CLI."""
    try:
        style_choice_raw = prompt_with_default(
            "Style (leave blank for none; options: pop, trip_hop, ambient, edm, funk, hiphop, dnb, liquid_dnb, electronic, experimental)",
            "",
        )
        style_choice = style_choice_raw.strip().lower() or None
        style_defaults = get_style_defaults(style_choice) if style_choice else {}

        key_input = input("Enter key (e.g., C, D#, F#): ").strip()
        mode_input = prompt_with_default(
            "Enter mode (major, minor, dorian, phrygian, lydian, mixolydian, aeolian, locrian)",
            str(style_defaults.get("mode", "ionian")),
        )
        bars = prompt_int("Number of bars (default 4): ", default=4, min_value=1)
        chord_complexity = min(
            prompt_int(
                "Chord complexity (1 = triads, 2 = 7ths, 3 = extended): ",
                default=int(style_defaults.get("chord_complexity", 2)),
                min_value=1,
            ),
            3,
        )
        progression_complexity = min(
            prompt_int(
                "Progression complexity (1 = simple pop, 2 = more movement, 3 = adventurous): ",
                default=int(style_defaults.get("progression_complexity", 1)),
                min_value=1,
            ),
            3,
        )
        bass_complexity = min(
            prompt_int(
                "Bass complexity (1 = roots, 2 = more movement): ",
                default=1,
                min_value=1,
            ),
            2,
        )
        bass_style = prompt_with_default(
            "Bass style (sustain, pulse, thump, movement, more_movement)",
            str(style_defaults.get("bass_style", "pulse")),
        )
        rhythm_style = min(
            prompt_int(
                "Chord rhythm style (1 = sustained, 2 = pulses, 3 = arpeggiated): ",
                default=int(style_defaults.get("rhythm_style", 1)),
                min_value=1,
            ),
            3,
        )
        humanization_amount = min(
            prompt_int(
                "Humanization amount (0 = none, 1 = slight, 2 = moderate): ",
                default=int(style_defaults.get("humanization_amount", 1)),
                min_value=0,
            ),
            2,
        )
        tempo_input = prompt_with_default("Tempo (BPM)", str(style_defaults.get("tempo", 100)))
        try:
            tempo_value = int(tempo_input)
        except ValueError:
            tempo_value = int(style_defaults.get("tempo", 100))

        use_borrowed = prompt_yes_no("Use borrowed/modal interchange chords? (y/n): ")
        filename_input = input('Output MIDI filename (default "progression.mid"): ').strip()
        folder_input = input("Folder to save into (default current directory): ").strip()

        filename = filename_input or "progression.mid"
        degree_labels, chord_names, output_path = generate_midi(
            key_input,
            mode_input,
            bars,
            chord_complexity,
            progression_complexity,
            bass_complexity,
            bass_style,
            rhythm_style,
            humanization_amount,
            use_borrowed,
            filename,
            folder_input,
            style_choice,
            tempo_override=tempo_value,
        )

        print(f"Style: {style_choice}")
        print(f"Scale ({normalize_mode(mode_input)}): {build_scale(key_input, normalize_mode(mode_input))[0]}")
        print(f"Progression (degrees): {degree_labels}")
        print(f"Progression (chords): {chord_names}")
        print(f"Created MIDI file: {output_path}")
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        sys.exit(1)


# -------------------- GUI -------------------- #
def run_gui() -> None:
    """Launch a tkinter GUI for MIDI generation with style and bass style controls."""
    keys = CHROMATIC
    modes = ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]
    styles = list(STYLE_PRESETS.keys())

    chord_complexity_opts = ["1 - Triads", "2 - 7ths", "3 - Extended"]
    progression_complexity_opts = ["1 - Simple", "2 - More movement", "3 - Adventurous"]
    rhythm_style_opts = ["1 - Sustained", "2 - Pulses", "3 - Arpeggiated"]
    humanization_opts = ["0 - None", "1 - Slight", "2 - Moderate"]
    bass_complexity_opts = ["1 - Roots", "2 - More movement"]
    bass_style_opts = ["sustain", "pulse", "thump", "movement", "more_movement"]

    root = tk.Tk()
    root.title("Modal Chord & Bassline Generator")
    root.geometry("640x600")
    root.minsize(620, 560)

    # Variables (must be created after root)
    style_var = tk.StringVar(master=root, value="pop")
    key_var = tk.StringVar(master=root, value="C")
    mode_var = tk.StringVar(master=root, value="major")
    bars_var = tk.StringVar(master=root, value="4")
    chord_complexity_var = tk.StringVar(master=root, value=chord_complexity_opts[1])
    prog_complexity_var = tk.StringVar(master=root, value=progression_complexity_opts[0])
    rhythm_style_var = tk.StringVar(master=root, value=rhythm_style_opts[1])
    humanization_var = tk.StringVar(master=root, value=humanization_opts[1])
    bass_complexity_var = tk.StringVar(master=root, value=bass_complexity_opts[0])
    bass_style_var = tk.StringVar(master=root, value="pulse")
    borrow_var = tk.BooleanVar(master=root, value=False)
    filename_var = tk.StringVar(master=root, value="progression.mid")
    folder_var = tk.StringVar(master=root, value=os.getcwd())
    tempo_var = tk.StringVar(master=root, value=str(get_style_defaults("pop").get("tempo", 100)))

    def parse_prefixed_int(value: str) -> int:
        try:
            return int(value.split(" ")[0])
        except Exception:
            raise ValueError(f"Could not parse numeric value from '{value}'.")

    def choose_folder():
        path = filedialog.askdirectory()
        if path:
            folder_var.set(path)

    def apply_style_defaults_to_fields(style_name: str):
        defaults = get_style_defaults(style_name)
        if defaults.get("mode"):
            mode_var.set(defaults["mode"])
        if defaults.get("chord_complexity"):
            chord_complexity_var.set(chord_complexity_opts[int(defaults["chord_complexity"]) - 1])
        if defaults.get("progression_complexity"):
            prog_complexity_var.set(progression_complexity_opts[int(defaults["progression_complexity"]) - 1])
        if defaults.get("rhythm_style"):
            rhythm_style_var.set(rhythm_style_opts[int(defaults["rhythm_style"]) - 1])
        if defaults.get("bass_style"):
            bass_style_var.set(defaults["bass_style"])
        if defaults.get("humanization_amount") is not None:
            humanization_var.set(humanization_opts[int(defaults["humanization_amount"])])
        if defaults.get("tempo"):
            tempo_var.set(str(defaults["tempo"]))

    def validate_inputs():
        key_val = key_var.get().strip()
        mode_val = mode_var.get().strip().lower()
        bars_val = bars_var.get().strip()
        filename_val = filename_var.get().strip()
        if not key_val or key_val not in keys:
            raise ValueError("Please choose a key from the list.")
        if mode_val not in modes:
            raise ValueError("Mode must be one of the provided options.")
        if not bars_val.isdigit() or int(bars_val) <= 0:
            raise ValueError("Bars must be a positive integer.")
        if not filename_val:
            raise ValueError("Please provide a filename.")
        return key_val, mode_val, int(bars_val), filename_val

    def update_result(degrees: List[str], chords: List[str], path: str) -> None:
        result_text.configure(state="normal")
        result_text.delete("1.0", tk.END)
        result_text.insert(tk.END, f"Degrees: {degrees}\nChords: {chords}\nSaved: {path}")
        result_text.configure(state="disabled")

    def generate():
        try:
            key_val, mode_val, bars_int, filename_val = validate_inputs()
            chord_c = parse_prefixed_int(chord_complexity_var.get())
            prog_c = parse_prefixed_int(prog_complexity_var.get())
            rhythm_c = parse_prefixed_int(rhythm_style_var.get())
            human_c = parse_prefixed_int(humanization_var.get())
            bass_complexity = parse_prefixed_int(bass_complexity_var.get())
            bass_style = bass_style_var.get()
            style_val = style_var.get().strip() or None
            tempo_val = tempo_var.get().strip()
            tempo_int = int(tempo_val) if tempo_val.isdigit() else get_style_defaults(style_val).get("tempo", 100)

            degree_labels, chord_names, output_path = generate_midi(
                key_val,
                mode_val,
                bars_int,
                chord_c,
                prog_c,
                bass_complexity,
                bass_style,
                rhythm_c,
                human_c,
                borrow_var.get(),
                filename_val,
                folder_var.get(),
                style_val,
                tempo_override=tempo_int,
            )
            update_result(degree_labels, chord_names, output_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", str(exc))

    apply_style_defaults_to_fields("pop")

    root.columnconfigure(0, weight=1)
    root.rowconfigure(3, weight=1)

    header = ttk.Label(root, text="Generates modal chord progressions and basslines to MIDI.", anchor="center")
    header.grid(row=0, column=0, pady=(10, 5), padx=12, sticky="we")

    content = ttk.Frame(root, padding=12)
    content.grid(row=1, column=0, sticky="nsew")
    content.columnconfigure(1, weight=1)

    # Style selection
    style_frame = ttk.Labelframe(content, text="Style Preset", padding=10)
    style_frame.grid(row=0, column=0, columnspan=2, sticky="we", pady=6)
    style_frame.columnconfigure(1, weight=1)
    ttk.Label(style_frame, text="Style:").grid(row=0, column=0, sticky="w", pady=2, padx=4)
    style_combo = ttk.Combobox(style_frame, textvariable=style_var, values=styles, state="readonly", width=18)
    style_combo.grid(row=0, column=1, sticky="we", pady=2, padx=4)

    def on_style_change(event=None):
        apply_style_defaults_to_fields(style_var.get())

    style_combo.bind("<<ComboboxSelected>>", on_style_change)

    # Musical Settings
    musical_frame = ttk.Labelframe(content, text="Musical Settings", padding=10)
    musical_frame.grid(row=1, column=0, columnspan=2, sticky="we", pady=6)
    musical_frame.columnconfigure(1, weight=1)
    ttk.Label(musical_frame, text="Key:").grid(row=0, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(musical_frame, textvariable=key_var, values=keys, state="readonly", width=10).grid(
        row=0, column=1, sticky="we", pady=2, padx=4
    )
    ttk.Label(musical_frame, text="Mode:").grid(row=1, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(musical_frame, textvariable=mode_var, values=modes, state="readonly", width=18).grid(
        row=1, column=1, sticky="we", pady=2, padx=4
    )
    ttk.Label(musical_frame, text="Bars:").grid(row=2, column=0, sticky="w", pady=2, padx=4)
    ttk.Entry(musical_frame, textvariable=bars_var, width=8).grid(row=2, column=1, sticky="w", pady=2, padx=4)
    ttk.Label(musical_frame, text="Tempo (BPM):").grid(row=3, column=0, sticky="w", pady=2, padx=4)
    ttk.Entry(musical_frame, textvariable=tempo_var, width=8).grid(row=3, column=1, sticky="w", pady=2, padx=4)

    # Chord Settings
    chord_frame = ttk.Labelframe(content, text="Chord Settings", padding=10)
    chord_frame.grid(row=2, column=0, sticky="nsew", pady=6, padx=(0, 6))
    chord_frame.columnconfigure(1, weight=1)
    ttk.Label(chord_frame, text="Chord complexity:").grid(row=0, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(chord_frame, textvariable=chord_complexity_var, values=chord_complexity_opts, state="readonly").grid(
        row=0, column=1, sticky="we", pady=2, padx=4
    )
    ttk.Label(chord_frame, text="Progression complexity:").grid(row=1, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(
        chord_frame, textvariable=prog_complexity_var, values=progression_complexity_opts, state="readonly"
    ).grid(row=1, column=1, sticky="we", pady=2, padx=4)
    ttk.Label(chord_frame, text="Chord rhythm style:").grid(row=2, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(chord_frame, textvariable=rhythm_style_var, values=rhythm_style_opts, state="readonly").grid(
        row=2, column=1, sticky="we", pady=2, padx=4
    )
    ttk.Checkbutton(chord_frame, text="Use borrowed/modal interchange", variable=borrow_var).grid(
        row=3, column=0, columnspan=2, sticky="w", pady=6, padx=4
    )

    # Bass & Humanization
    bass_frame = ttk.Labelframe(content, text="Bass & Humanization", padding=10)
    bass_frame.grid(row=2, column=1, sticky="nsew", pady=6, padx=(6, 0))
    bass_frame.columnconfigure(1, weight=1)
    ttk.Label(bass_frame, text="Bass complexity:").grid(row=0, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(bass_frame, textvariable=bass_complexity_var, values=bass_complexity_opts, state="readonly").grid(
        row=0, column=1, sticky="we", pady=2, padx=4
    )
    ttk.Label(bass_frame, text="Bass style:").grid(row=1, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(bass_frame, textvariable=bass_style_var, values=bass_style_opts, state="readonly").grid(
        row=1, column=1, sticky="we", pady=2, padx=4
    )
    ttk.Label(bass_frame, text="Humanization:").grid(row=2, column=0, sticky="w", pady=2, padx=4)
    ttk.Combobox(bass_frame, textvariable=humanization_var, values=humanization_opts, state="readonly").grid(
        row=2, column=1, sticky="we", pady=2, padx=4
    )

    # Output
    output_frame = ttk.Labelframe(content, text="Output", padding=10)
    output_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", pady=6)
    output_frame.columnconfigure(1, weight=1)
    output_frame.columnconfigure(2, weight=1)
    ttk.Label(output_frame, text="Filename:").grid(row=0, column=0, sticky="w", pady=2, padx=4)
    ttk.Entry(output_frame, textvariable=filename_var, width=22).grid(row=0, column=1, sticky="we", pady=2, padx=4)
    ttk.Label(output_frame, text="Folder:").grid(row=1, column=0, sticky="w", pady=2, padx=4)
    ttk.Entry(output_frame, textvariable=folder_var).grid(row=1, column=1, sticky="we", pady=2, padx=4)
    ttk.Button(output_frame, text="Browse...", command=choose_folder).grid(row=1, column=2, sticky="e", padx=4)
    button_row = ttk.Frame(output_frame)
    button_row.grid(row=2, column=0, columnspan=3, sticky="e", pady=(8, 4))
    ttk.Button(button_row, text="Generate MIDI", command=generate).pack(side="right", padx=4)
    ttk.Button(button_row, text="Close", command=root.destroy).pack(side="right", padx=4)

    # Result display
    result_frame = ttk.Labelframe(content, text="Result", padding=10)
    result_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=6)
    result_frame.columnconfigure(0, weight=1)
    result_frame.rowconfigure(0, weight=1)
    result_text = tk.Text(result_frame, height=6, wrap="word", state="disabled")
    result_text.grid(row=0, column=0, sticky="nsew")
    scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=result_text.yview)
    scrollbar.grid(row=0, column=1, sticky="ns")
    result_text.configure(yscrollcommand=scrollbar.set)

    root.mainloop()


if __name__ == "__main__":
    # If the user passes "--cli", run the terminal version.
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        main()
    else:
        run_gui()
