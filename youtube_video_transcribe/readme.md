# YouTube Video Transcription & Summarization

This project extracts transcripts from YouTube videos and generates concise summaries using Google's Gemini AI. 

## Test Case
Tested on MrBeast's celebrity competition video: https://youtu.be/QJI0an6irrA?si=Jr5_Lv3x6WnH0Fnn

**Generated Summary:**
> MrBeast hosts a celebrity competition where a large group of famous individuals, including Kevin Hart, Paris Hilton, Howie Mandel, and musical acts like Five Seconds of Summer and The Chainsmokers, compete for a $1 million prize for their chosen charities. Participants face multiple elimination rounds: Dodgeball, Smart vs. Strong challenges, Cooking Challenge, and Cube Eliminations. The final three competitors are Audrey Nuna, Yung Gravy, and Steve-O. Steve-O wins the $1 million for his charity, Doctors Without Borders. The video also features promotions for MrBeast's "Beast Games Season 2" on Prime Video.

## Implementation

**Requirements:**
```bash
pip install youtube_transcript_api google-generativeai
```

**Key Steps:**
1. Extract video ID from YouTube URL
2. Fetch transcript using YouTube Transcript API
3. Join transcript segments into single text
4. Send to Google Gemini 2.5 Flash for summarization

**Technologies:**
- YouTube Transcript API
- Google Gemini 2.5 Flash
- Python/Jupyter Notebook

**Note:** Works best in Google Colab to avoid IP blocking issues.
