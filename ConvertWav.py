from moviepy import VideoFileClip

video = VideoFileClip("fortnite.mp4")
audio = video.audio
audio.write_audiofile("music.wav")