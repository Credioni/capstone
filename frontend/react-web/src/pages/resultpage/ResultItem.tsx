import React, { useState } from 'react';
import { Card, Link, Divider, CardMedia, Box, Tooltip, Collapse, IconButton } from '@mui/material';
import { Typography, List, ListItem, ListItemText } from "@mui/material";
import AutoAwesomeMotionIcon from '@mui/icons-material/AutoAwesomeMotion';
import MediaPreview from "./MediaPreview"
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import ExpandLessIcon from '@mui/icons-material/ExpandLess';
import ImageIcon from '@mui/icons-material/Image';
import AudioFileIcon from '@mui/icons-material/AudioFile';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import ArticleIcon from '@mui/icons-material/Article';
import FD from "../../assets/FD_e_mu.jpg"
import Sound from "../../assets/arcade.wav"
import PDFdummy from "../../assets/2503.08254v1.pdf"


const DUMMYrightPanelData = {
    title: "Warhammer 40,000 Commander (40K)",
    sections: [
      {
        title: "Pictures",
        items: [
          {
            key: "Plot Quantum field Theory Graph",
            type: "image",
            value: FD,
          },
        ],
      },
      {
        title: "Pdf Files",
        items: [
          {
            key: "Pdf",
            type: "pdf",
            value: PDFdummy,
          },
        ]
      },
      {
        title: "Audio Files",
        items: [
          {
            key: "Music Track",
            type: "audio",
            value: Sound,
          },
        ]
      },
    ],
    searchInfo: [
        {text: "Temperature", value:"0.1"},
        {text: "ABC", value:"0.1"}
    ],
  };

/**
 * InfoSection component displays a section with media items
 */
const InfoSection = ({ title, items, onItemHover }) => {
  // Get the appropriate icon based on media type
  const getItemIcon = (type) => {
    switch (type) {
      case 'image':
        return <ImageIcon fontSize="small" sx={{ color: '#FFD700' }} />;
      case 'audio':
        return <AudioFileIcon fontSize="small" sx={{ color: '#FFD700' }} />;
      case 'pdf':
        return <PictureAsPdfIcon fontSize="small" sx={{ color: '#FFD700' }} />;
      default:
        return <ArticleIcon fontSize="small" sx={{ color: '#FFD700' }} />;
    }
  };

  return (
    <div className="mb-3">
      {/* Section header */}
      {/* <div className="font-medium text-white py-1 px-2 bg-[#343243]">
        {title}
      </div> */}

      {/* Section content */}
      <List dense disablePadding>
        {items.map((item, idx) => (

            <ListItem
                key={idx}
              className="pl-4 hover:bg-gray-100 cursor-pointer transition-colors"
              onMouseEnter={(event) => onItemHover(item, event)}
              onMouseMove={(event) => onItemHover(item, event)}
              onClick={(event) => onItemHover(item, event)}
              sx={{ borderBottom: '1px solid #e0e0e0' }}
            >
              <div className="mr-2">
                {getItemIcon(item.type)}
              </div>
              <ListItemText
                primary={item.key}
                primaryTypographyProps={{
                  className: "text-sm font-medium text-gray-800"
                }}
              />
            </ListItem>
        ))}
      </List>
    </div>
  );
};

function ResultItemInfo({ info, ...args }) {
    const [hoveredItem, setHoveredItem] = useState(null);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

    const handlePreviewChange = (item, event) => {
        setHoveredItem(item);
        // Get mouse position relative to the viewport
        setMousePosition({
            x: event.clientX,
            y: event.clientY
        });
    };

    const handleMouseLeave = () => {
        setHoveredItem(null);
    };

    return (
        <div
            className="grid-flow-col m-2 relative"
            {...args}
            onMouseLeave={handleMouseLeave}
        >
            {/* Top title */}
            <div className="bg-[#343243] rounded-t-md p-2">
                <div className="flex">
                    <AutoAwesomeMotionIcon sx={{ color: 'white' }}/>

                    <div className="ml-2 text-white text-md">
                        Article Content Embedded
                    </div>
                </div>
                <hr className="border-2 border-solid border-gray-600 w-[85%] ml-[14%]"/>
            </div>

            <List dense>
                {info.sections.map((item, idx) => (
                    <InfoSection
                        key={idx}
                        title={item.title}
                        items={item.items}
                        onItemHover={(hoveredItem, event) => handlePreviewChange(hoveredItem, event)}
                    />
                ))}
            </List>

            {/* Preview window that appears when hovering */}
            {hoveredItem && (
                <MediaPreview
                    item={hoveredItem}
                    position={mousePosition}
                />
            )}
        </div>
    );
};

function ResultCardTitle({doi, pdf_link=null, ...args}) {
    return (
    <Typography {...args} sx={{ display: 'flex', flexDirection: 'row' }}>
        <Link href={doi} className='pr-4'> {doi}</Link>
        [
            <Link href={pdf_link || "#"}>pdf</Link>,
            <Link href={pdf_link || "#"}>other</Link>
        ]
    </Typography>
    );
}

function ResultCard({index, result, ...args }) {
  const [hoveredItem, setHoveredItem] = useState(null);

  return (
    <Card className='flex flex-col border-black border-5' {...args} >
        {/* Top title */}
        <ResultCardTitle className="pl-4" index={index} doi={result.id} />

        {/* Main content */}
        <div className="flex pl-4 pb-4">
            {/* Article Information */}
            <div className="flex-1 w-8/12 pr-4">
                <Typography gutterBottom variant="h6" component="div">
                    { result.title }
                </Typography>
                <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    { result.abstract }
                </Typography>
            </div>

            {/* Article Multimedia Preview - Fixed width container */}
            {/* <div className="w-4/12" style={{ minWidth: '200px' }}>
                <Divider orientation='vertical' sx={{ position: 'absolute', height: '80%' }} />
                <Box sx={{ pl: 2 }}>
                <MediaPreview item={hoveredItem} />
                </Box>
            </div> */}

            {/* Right Bar Content Info */}
            <ResultItemInfo
                className='ml-5 mr-2 w-4/12 max-w-80'
                info={DUMMYrightPanelData}
                onPreviewChange={setHoveredItem}
            />
        </div>
    </Card>
  );
}


export default function ResultItem({index, result, ...args}) {
    const authorText = result.authors ? result.authors.join(', ') : '';

    return (
        <ListItem sx={{ display: "list-item" }}>
            <ResultCard index={index} result={result} {...args}/>
      </ListItem>
    );
}