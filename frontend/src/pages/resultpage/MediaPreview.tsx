import React from 'react';
import { Paper, Typography } from '@mui/material';
import CloseIcon from '@mui/icons-material/Close';

/**
 * MediaPreview component displays different types of media
 * based on the hovered item
 */
const MediaPreview = ({ item, position, onClose=null, isFrozen=false }) => {
  if (!item) return null;

  return (
        /* Render content */
        <Paper
            elevation={3}
            className="z-50 rounded-md overflow-hidden p-1"
            sx={{
            width: '300px',
            minHeight: 'fit',
            maxHeight: '400px',
            border: '1px solid #333',
            position: 'fixed',
            left: `${position.x + 40}px`, // Position to the left of cursor
            top: `${position.y}px`         // Top corner at cursor position
            }}
        >
            {isFrozen && (
                <button
                    onClick={onClose}
                    className="text-black hover:text-red-300 focus:outline-none"
                    style={{fontSize: '18px', fontWeight: 'bold', lineHeight: 1 }}
                >
                    <CloseIcon/>
                </button>
            )}

            {renderMediaContent(item)}
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
    // case 'pdf':
    //   return (
    //     <div className="w-full h-[350px]">
    //       <object
    //         data={item.value}
    //         type="application/pdf"
    //         width="100%"
    //         height="100%"
    //       >
    //         <Typography>
    //           PDF preview not available. <a href={item.value} target="_blank" rel="noopener noreferrer">Open PDF</a>
    //         </Typography>
    //       </object>
    //     </div>
    //   );

    case 'audio':
      return (
        <div className="w-full text-center">
          {/* <img
            src="/api/placeholder/200/120"
            alt=""
            className="mb-2"
          /> */}
          <audio
            // autoPlay
            controls
            className="w-full"
        >
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