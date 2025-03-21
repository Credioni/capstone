import React, { useRef, useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  IconButton,
  Slider
} from '@mui/material';
import HeadphonesIcon from '@mui/icons-material/Headphones';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import PauseIcon from '@mui/icons-material/Pause';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import VolumeOffIcon from '@mui/icons-material/VolumeOff';

const AudioDisplay = ({ audioFiles }) => {
    // If no audio files are provided, return nothing
    if (!audioFiles || audioFiles.length === 0) {
        return null;
    }

    return (
        <Box sx={{ width: '100%' }}>
        <Typography variant="h6" sx={{ mb: 2, textAlign: 'center' }}>
            Audio Files
        </Typography>
        <Box sx={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            gap: 2
        }}>
            {audioFiles.map((audio, index) => (
            <AudioCard key={index} audio={audio} index={index} />
            ))}
        </Box>
        </Box>
    );
};

// Separate component for each audio card with custom controls
const AudioCard = ({ audio, index }) => {
  const audioRef = useRef(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Handle audio metadata load to get duration
  useEffect(() => {
    const handleLoadedMetadata = () => {
      if (audioRef.current) {
        setDuration(audioRef.current.duration);
      }
    };

    // Update time as audio plays
    const handleTimeUpdate = () => {
      if (audioRef.current) {
        setCurrentTime(audioRef.current.currentTime);
      }
    };

    const audioElement = audioRef.current;
    if (audioElement) {
      audioElement.addEventListener('loadedmetadata', handleLoadedMetadata);
      audioElement.addEventListener('timeupdate', handleTimeUpdate);
      audioElement.addEventListener('ended', () => setIsPlaying(false));
    }

    return () => {
      if (audioElement) {
        audioElement.removeEventListener('loadedmetadata', handleLoadedMetadata);
        audioElement.removeEventListener('timeupdate', handleTimeUpdate);
        audioElement.removeEventListener('ended', () => setIsPlaying(false));
      }
    };
  }, []);

  // Play/pause toggle
  const togglePlay = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
      } else {
        audioRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  // Toggle mute
  const toggleMute = () => {
    if (audioRef.current) {
      audioRef.current.muted = !audioRef.current.muted;
      setIsMuted(!isMuted);
    }
  };

  // Handle slider change
  const handleSliderChange = (_, newValue) => {
    if (audioRef.current) {
      audioRef.current.currentTime = newValue;
      setCurrentTime(newValue);
    }
  };

  // Format time (seconds to MM:SS)
  const formatTime = (time) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds < 10 ? '0' : ''}${seconds}`;
  };

  return (
    <Card sx={{ width: 280, height: 'auto', m: 1 }}>
        <CardContent>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
            <Typography
                variant="subtitle1"
                noWrap sx={{ maxWidth: 200 }}
                className='justify-self-center'
            >
                {audio.id ? audio.id.split('.')[0] : `Audio ${index + 1}`}
            </Typography>

            <Typography variant="caption" color="text.secondary">
                {duration ? formatTime(duration) : '--:--'}
            </Typography>
            </Box>

        <Box sx={{ display: 'flex', justifyContent: 'center', my: 2 }}>
            <Box sx={{
                backgroundColor: 'primary.light',
                borderRadius: '50%',
                p: 2,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
            }}>
            <HeadphonesIcon sx={{ fontSize: 40, color: 'primary.main' }} />
            </Box>
        </Box>

        {/* Hidden native audio element */}
        <audio
            ref={audioRef}
            src={`data:${audio.mime_type || 'audio/wav'};base64,${audio.base64}`}
            preload="metadata"
            style={{ display: 'none' }}
        />

        {/* Custom audio controls */}
        <Box sx={{ width: '100%', mb: 1 }}>
          {/* Time display and progress slider */}
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
            <Typography variant="caption" sx={{ mr: 1, minWidth: '35px' }}>
              {formatTime(currentTime)}
            </Typography>
            <Slider
              size="small"
              value={currentTime}
              max={duration || 100}
              onChange={handleSliderChange}
              aria-label="Audio progress"
              sx={{ mx: 1 }}
            />
            <Typography variant="caption" sx={{ ml: 1, minWidth: '35px' }}>
              {formatTime(duration)}
            </Typography>
          </Box>

          {/* Play/pause and volume controls */}
          <Box sx={{ display: 'flex', mt: 1, justifyItems: "center", justifySelf: "center" }}>
            <IconButton onClick={togglePlay} size="small">
              {isPlaying ? <PauseIcon /> : <PlayArrowIcon />}
            </IconButton>
            <IconButton onClick={toggleMute} size="small">
              {isMuted ? <VolumeOffIcon /> : <VolumeUpIcon />}
            </IconButton>
          </Box>

          <Typography variant='body2' className='justify-items-center justify-self-center'>
            {"Faiss score: " + audio.score.toFixed(3)}
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default AudioDisplay;