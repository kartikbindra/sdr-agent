import React, { useState, useRef } from 'react';

export default function VoiceSDRAgent() {
  const [recording, setRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [audioSrc, setAudioSrc] = useState('');
  const [mediaElement, setMediaElement] = useState(null);
  const [leadInfo, setLeadInfo] = useState(null);
  const socketRef = useRef(null);
  const mediaRecorderRef = useRef(null);

  const handleStart = async () => {
    socketRef.current = new WebSocket('ws://localhost:8000/ws');

    socketRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'agent_response') {
        setTranscript(data.text);
        setAudioSrc(data.audio_url);

        if (data.media) {
          const media =
            data.media.type === 'video' ? (
              <video src={data.media.url} controls width="400" />
            ) : (
              <img src={data.media.url} width="400" alt="Media content" />
            );
          setMediaElement(media);
        }

        if (data.lead) {
          setLeadInfo(data.lead);
        }
      }
    };

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mediaRecorder = new MediaRecorder(stream);
    mediaRecorderRef.current = mediaRecorder;
    mediaRecorder.start();
    setRecording(true);

    mediaRecorder.ondataavailable = (e) => {
      const reader = new FileReader();
      reader.readAsDataURL(e.data);
      reader.onloadend = () => {
        const base64data = reader.result.split(',')[1];
        socketRef.current.send(JSON.stringify({ type: 'audio', data: base64data }));
      };
    };
  };

  const handleStop = () => {
    mediaRecorderRef.current.stop();
    setRecording(false);
  };

  return (
    <div className="p-4 max-w-xl mx-auto">
      <h1 className="text-xl font-bold mb-4">AI Voice SDR Agent</h1>
      <div className="flex gap-4 mb-4">
        <button
          onClick={handleStart}
          disabled={recording}
          className="bg-blue-500 text-white px-4 py-2 rounded"
        >
          Start Recording
        </button>
        <button
          onClick={handleStop}
          disabled={!recording}
          className="bg-red-500 text-white px-4 py-2 rounded"
        >
          Stop
        </button>
      </div>
      <p className="text-lg font-medium mb-2">Agent Response:</p>
      <p className="bg-gray-100 p-2 rounded mb-4">{transcript}</p>
      {audioSrc && <audio src={audioSrc} controls className="mb-4" />}
      {mediaElement && <div className="mb-4">{mediaElement}</div>}
      {leadInfo && (
        <div className="bg-yellow-100 p-4 rounded">
          <h2 className="font-semibold">Lead Info</h2>
          <pre className="text-sm whitespace-pre-wrap">{JSON.stringify(leadInfo, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}
