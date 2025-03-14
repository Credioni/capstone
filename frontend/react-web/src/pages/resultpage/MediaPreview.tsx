import React from 'react';
import { Paper, Typography } from '@mui/material';

/**
 * MediaPreview component displays different types of media
 * based on the hovered item
 */
const MediaPreview = ({ item, position }) => {
  if (!item) return null;

  return (
    <Paper
      elevation={3}
      className="z-50 rounded-md overflow-hidden"
      sx={{
        width: '300px',
        minHeight: '200px',
        maxHeight: '400px',
        border: '1px solid #333',
        position: 'fixed',
        left: `${position.x + 40}px`, // Position to the left of cursor
        top: `${position.y}px`         // Top corner at cursor position
      }}
    >
{renderMediaContent(item)}
      {/* <div className="bg-[#343243] p-2 text-white">
        <Typography variant="subtitle2">{item.key}</Typography>
      </div>

      <div className="flex justify-center items-center p-2 bg-white h-full">
        {renderMediaContent(item)}
      </div> */}
    </Paper>
  );
};

/**
 * Renders the appropriate media content based on item type
 */
const renderMediaContent = (item) => {
  switch (item.type) {
    case 'image':
      return (
        <img
          src={item.value}
          alt={item.key}
          className="max-w-full max-h-[350px] object-contain"
        />
      );

    case 'pdf':
      return (
        <div className="w-full h-[350px]">
          <object
            data={item.value}
            type="application/pdf"
            width="100%"
            height="100%"
          >
            <Typography>
              PDF preview not available. <a href={item.value} target="_blank" rel="noopener noreferrer">Open PDF</a>
            </Typography>
          </object>
        </div>
      );

    case 'audio':
      return (
        <div className="w-full text-center">
          {/* <img
            src="/api/placeholder/200/120"
            alt=""
            className="mb-2"
          /> */}
          <audio autoPlay controls className="w-full">
                <source src={item.value} />
                Your browser does not support the audio element.
          </audio>
        </div>
      );

    default:
      return (
        <Typography>
          Preview not available for this content type.
        </Typography>
      );
  }
};

export default MediaPreview;