import spacy as spacy
from ffmpy import FFmpeg
import time
import azure.cognitiveservices.speech as speechsdk
from tkinter import filedialog
import tkinter as tk
import os
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np


def video_audio(video_name, audio_name):
    ff = FFmpeg(inputs={video_name: None}, outputs={audio_name: None})
    ff.run()


def speech_recognize_continuous_from_file(audio_file, f):
    """performs continuous speech recognition with input from an audio file"""
    # <SpeechContinuousRecognitionWithFile>
    speech_config = speechsdk.SpeechConfig(subscription="c50c491e9cb64d788c4412d8d204aa7e", region="eastus")
    # speech_config.request_word_level_timestamps()
    audio_config = speechsdk.audio.AudioConfig(filename=audio_file)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event `evt`"""
        print('CLOSING on {}'.format(evt))
        nonlocal done
        done = True

    duration = []
    text = []
    offset = []

    def handle_final_result(evt):
        text.append(evt.result.text)
        offset.append(evt.result.offset)
        duration.append(evt.result.duration)

    speech_recognizer.recognized.connect(handle_final_result)
    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognizing.connect(lambda evt: print('RECOGNIZING: {}'.format(evt)))
    speech_recognizer.recognized.connect(lambda evt: print('RECOGNIZED: {}'.format(evt.result)))
    speech_recognizer.session_started.connect(lambda evt: print('SESSION STARTED: {}'.format(evt)))
    speech_recognizer.session_stopped.connect(lambda evt: print('SESSION STOPPED {}'.format(evt)))
    speech_recognizer.canceled.connect(lambda evt: print('CANCELED {}'.format(evt)))
    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # Start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(.5)
    for i in text:
        f.write(i)
    return duration, text, offset


def topic_slot(arr, text, filtered_keywords):
    # Find the frequency of keywords
    time_freq = []
    for j in range(len(arr)):
        count = 0
        for k in range(len(text)):
            if arr[j][k] != 0.0:
                count = count + 1
        time_freq.append(count)

    # Find the average time of keyword
    average_time = []
    for i in range(len(arr)):
        av = 0
        num = 0
        total = 0
        while arr[i][num] != 0.0:
            total = total + arr[i][num]
            num = num + 1

        av = total / num
        average_time.append(av)

    time_slot = [[0.0] * len(average_time) for _ in range(len(average_time))]
    key_slot = [[0.0] * len(average_time) for _ in range(len(average_time))]
    visited = []
    for i in range(len(average_time)):
        visited.append(0)

    time_slot[0][0] = average_time[0]
    key_slot[0][0] = filtered_keywords[0]
    visited[0] = 1
    # Group keyword into groups according to their average time
    for i in range(len(average_time)):
        count = 0
        for j in range(len(average_time)):
            if time_slot[i][j] != 0.0:
                for k in range(len(average_time)):
                    diff = average_time[k] - time_slot[i][j]
                    if 3 >= diff >= -3 and visited[k] != 1:
                        count = count + 1
                        time_slot[i][count] = average_time[k]
                        key_slot[i][count] = filtered_keywords[k]
                        visited[k] = 1
            else:
                for k in range(len(average_time)):
                    if visited[k] == 0:
                        time_slot[i + 1][0] = average_time[k]
                        key_slot[i + 1][0] = filtered_keywords[k]
                        visited[k] = 1
                        break
                break

    # Organize the group
    for i in range(len(average_time)):
        temp = []
        key_temp = []
        count = 0
        while time_slot[i][count] != 0:
            temp.append(time_slot[i][count])
            key_temp.append(key_slot[i][count])
            count = count + 1

        for m in range(len(temp)):
            for n in range(0, len(temp) - m - 1):
                if temp[n] > temp[n + 1]:
                    tmp = temp[n]
                    temp[n] = temp[n + 1]
                    temp[n + 1] = tmp
                    key_tmp = key_temp[n]
                    key_temp[n] = key_temp[n + 1]
                    key_temp[n + 1] = key_tmp
            num = 0
        while time_slot[i][num] != 0:
            time_slot[i][num] = temp[num]
            key_slot[i][num] = key_temp[num]
            num = num + 1

    for m in range(len(average_time)):
        for n in range(0, len(average_time) - m - 1):
            if time_slot[n][0] > time_slot[n + 1][0]:
                tmp = time_slot[n]
                time_slot[n] = time_slot[n + 1]
                time_slot[n + 1] = tmp
                key_tmp = key_slot[n]
                key_slot[n] = key_slot[n + 1]
                key_slot[n + 1] = key_tmp

    final = []
    key_fianl = []

    for m in range(len(average_time)):
        if time_slot[m][0] != 0.0:
            final.append(time_slot[m])
            key_fianl.append(key_slot[m])

    slot = [[0] * 2 for _ in range(len(final))]
    time_diff = []

    # Find the duration of each time slot
    for i in range(len(final) - 1):
        num = 0
        tmp = 0
        while final[i][num] != 0.0:
            tmp = final[i + 1][0] - final[i][num]
            num = num + 1
        tmp = tmp / 2
        tmp = final[i][num - 1] + tmp
        time_diff.append(tmp)

    slot[0][0] = 0
    slot[len(slot) - 1][1] = 90

    for i in range(len(slot) - 1):
        slot[i][1] = time_diff[i]
        slot[i + 1][0] = time_diff[i]

    return slot, final, key_fianl, time_freq


def graph(arr, text, filtered_keywords):
    y_list = []
    x_list = []
    # Change 2D List to 1D, for the graph
    # X axis is the keyword, and Y axis is the time
    count = 0
    for r in arr:
        for i in range(len(text)):
            if r[i] == 0.0:
                break
            else:
                x_list.append(count)

                y_list.append(r[i])
        count = count + 1
    dd = []
    count = 0
    for i in range(len(filtered_keywords)):
        dd.append(count)
        count = count + 1
    x = np.array(x_list)
    y = np.array(y_list)
    plt.xticks(dd, filtered_keywords)
    plt.scatter(x, y)
    plt.show()


def display_topic(slot, filtered_keywords, final, key_fianl, time_freq, arr, text):
    f = open('topic.txt', 'w')
    k = 0
    for i in range(len(slot)):
        print(slot[i][0], " to ", slot[i][1], ":")
        f.write(str(slot[i][0]) + " to " + str(slot[i][1]) + ":\n")
        num = 0
        # If the keyword has one appearance in the time slot, put it in that time slot
        '''for j in range(len(arr)):
            for k in range(len(text)):
                if slot[i][1] >= arr[j][k] >= slot[i][0] and arr[j][k] != 0.0:
                    print(filtered_keywords[j])
                    f.write(str(filtered_keywords[j]) + "\n")
                    break'''
        # If the average time of keyword is in the time slot, put it in that time slot
        '''while final[i][num] != 0.0:
            print(key_fianl[i][num])
            f.write(str(key_fianl[i][num]) + "\n")
            num = num + 1'''
        # For the word that has a frequency less than 20 and it's average time is with the time slot, put it in that
        # time slot. Also for the word that has a frequency larger than 20 the keyword has one appearance in the time
        # slot, put it in that time slot
        while final[i][num] != 0.0:
            for m in range(len(filtered_keywords)):
                if key_fianl[i][num] == filtered_keywords[m] and time_freq[m] < 20:
                    f.write(str(key_fianl[i][num]) + "\n")
            num = num + 1
        for j in range(len(arr)):
            for k in range(len(text)):
                if time_freq[j] >= 20 and slot[i][1] >= arr[j][k] >= slot[i][0] and arr[j][k] != 0.0:
                    f.write(str(filtered_keywords[j]) + "\n")
                    break
    # Dis play the video segmentation in the box
    k = 0
    with open('topic.txt') as f:
        for line in f:
            topic_label.insert(k, line)
            k = k + 1


def filename():
    # Open the file
    root.filename = filedialog.askopenfilename(title="Select file",
                                               filetypes=(
                                                   ("video files", "*.mp4 *.avi *.wmv *.mkv"), ("all files", "*.*")))
    # Get the audio and text file name
    audio = root.filename.rsplit('.', 1)[0]
    audio = audio.rsplit('/')[-1]
    text_file = audio + '.txt'
    audio_file = audio + '.wav'
    # Convert Video to Audio
    video_audio(root.filename, audio_file)
    f = open(text_file, "w")
    # Audio Video to Text
    duration, text, offset = speech_recognize_continuous_from_file(audio_file, f)
    # Remove Audio File
    os.remove(audio_file)
    nlp = spacy.load("td_ner_model")
    file = open(text_file, "r")

    line = file.read().replace("\n", " ")

    word = nlp(line)
    a = []
    for ent in word.ents:
        a.append(ent.text)

    a = list(set(a))
    # filter some useless word
    filtered_keywords = []

    for word in a:
        if word.isnumeric() != False or word.find("?") != -1 or word.find("!") != -1 or word.find("#") != -1 \
                or word.find("'") != -1 or ("." in word) or ("to" in word) or ('/' in word) or ("during" in word):
            continue
        filtered_keywords.append(word)

    # Display Keywords in the box
    k = 1
    for i in filtered_keywords:
        keyword_label.insert(k, i)
        k = k + 1
    # Starting time of each sentence
    o = []
    for x in offset:
        num = x / 600000000
        o.append(num)
    arr = [[0.0] * len(text) for _ in range(len(filtered_keywords))]
    # 2D List for the time of each keyword
    row = 0
    for x in range(len(filtered_keywords)):
        col = 0
        for i in range(len(text)):
            if filtered_keywords[x] in text[i]:
                arr[row][col] = o[i]
                col = col + 1
        row = row + 1

    slot, final, key_fianl, time_freq = topic_slot(arr, text, filtered_keywords)
    display_topic(slot, filtered_keywords, final, key_fianl, time_freq, arr, text)
    graph(arr, text, filtered_keywords)


root = tk.Tk()
background_image = tk.PhotoImage(file="picture.PNG")
background_label = tk.Label(root, image=background_image)
background_label.place(relx=0.5, rely=0.0, anchor='n')
root.geometry("750x750")
root.title("Video Segmentation")

frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.2, relwidth=0.2, relheight=0.1, anchor='n')

button = tk.Button(frame, text="Choose the file", command=filename, fg='yellow', bg='light blue')
button.place(relx=0, rely=0, relwidth=1, relheight=1)

keyword_result = tk.Frame(root)
keyword_result.place(relx=0.5, rely=0.35, relwidth=0.7, relheight=0.3, anchor='n')

keyword_label = tk.Listbox(keyword_result)
keyword_scrollbar = Scrollbar(keyword_result)
keyword_scrollbar.pack(side=RIGHT, fill=Y)
keyword_label.pack(fill=BOTH, expand=True)
keyword_label.config(yscrollcommand=keyword_scrollbar.set)
keyword_scrollbar.config(command=keyword_label.yview)

topic_result = tk.Frame(root)
topic_result.place(relx=0.5, rely=0.68, relwidth=0.7, relheight=0.3, anchor='n')

topic_label = tk.Listbox(topic_result)
topic_scrollbar = Scrollbar(topic_result)
topic_scrollbar.pack(side=RIGHT, fill=Y)
topic_label.pack(fill=BOTH, expand=True)
topic_label.config(yscrollcommand=topic_scrollbar.set)
topic_scrollbar.config(command=topic_label.yview)

root.mainloop()