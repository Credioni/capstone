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

interface ResultArxiv {
    id: string;
    score: number;
    text: string;
    title: string
}

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
        title: "Audio Files",
        items: [
          {
            key: "Sound of Gravitation Waves of Black Hole",
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

function formDOI(num) {
    // Convert number to string and split at the decimal point
    let [integerPart, decimalPart] = num.toString().split(".");
    integerPart = integerPart.padStart(4, "0");
    // Pad the decimal part with trailing zeros if necessary, ensuring 4 digits
    decimalPart = (decimalPart || "").padEnd(4, "0");
    return `${integerPart}.${decimalPart}`;
}

/**
 * InfoSection component displays a section with media items
 */
const InfoSection = ({ title, items, onItemHover, onItemClick }) => {
    // Get the appropriate icon based on media type
    const getItemIcon = (type) => {
        switch (type) {
            case 'image':
                return <ImageIcon fontSize="small" sx={{ color: '#343243' }} />;
            case 'audio':
                return <AudioFileIcon fontSize="small" sx={{ color: '#343243' }} />;
            case 'pdf':
                return <PictureAsPdfIcon fontSize="small" sx={{ color: '#343243' }} />;
            default:
                return <ArticleIcon fontSize="small" sx={{ color: '#343243' }} />;
        }
    };

    return (
        <div className='border-l-2 border-[#34324335]'>
            {/* Section header */}
            {/* <div className="font-medium text-white py-1 px-2 bg-[#343243]">
            {title}
            </div> */}

            {/* Section content */}
            <List dense disablePadding>
                {items.map((item, idx) => (
                    <ListItem dense disablePadding
                        key={idx}
                        className="hover:bg-gray-300 cursor-pointer transition-colors"
                        onMouseEnter={(event) => onItemHover(item, event)}
                        onMouseMove={(event) => onItemHover(item, event)}
                        onClick={(event) => onItemClick(item, event)}
                        // sx={{ borderBottom: '1px solid #e0e0e0' }}
                    >
                        <div className="mr-2">
                            {getItemIcon(item.type)}
                        </div>
                        <ListItemText
                            primary={item.key}
                            // primaryTypographyProps={{
                            //     className: "text-sm font-medium text-gray-800"
                            // }}
                        />
                    </ListItem>
                ))}
            </List>
        </div>
    );
};

function ResultItemInfo({ info, ...args }) {
    const DUMMY_CONTENT = [
        "Plotted Relevant information",
        "Temperature as an function of Score",
        "Deepthink paragraph",
        "Paragraph",
    ];

    const [previewItem, setPreviewItem] = useState(null);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const [isPreviewFrozen, setIsPreviewFrozen] = useState(false);

    const handlePreviewChange = (item, event) => {
        if (isPreviewFrozen) { return }
        setPreviewItem(item);
        // Get mouse position relative to the viewport
        setMousePosition({
            x: event.clientX,
            y: event.clientY
        });
    };

    const onClose = (item, event) => {
        setIsPreviewFrozen(false);
        handleMouseLeave();
    };

    const handleItemClick = (item, event) => {
        // Freeze the preview at current position
        handlePreviewChange(item, event);
        setIsPreviewFrozen(!isPreviewFrozen);
    };

    const handleMouseLeave = () => {
        if (!isPreviewFrozen){
            setPreviewItem(null);
        }
    };

    return (
        <div
            className="grid-flow-col m-2 relative"
            onMouseLeave={handleMouseLeave}
            {...args}
        >
            {/* Top title */}
            <div className="bg-[#343243] rounded-t-md p-2">
                <div className="flex">
                    <AutoAwesomeMotionIcon sx={{ color: 'white' }}/>

                    <div className="ml-2 text-white text-md">
                        Article Content
                    </div>
                </div>
                <hr className="border-l-2 border-solid border-gray-600 w-[85%] ml-[10%]"/>
            </div>

            <List dense onMouseLeave={handleMouseLeave}>
                {info.sections.map((item, idx) => (
                    <InfoSection
                        key={idx}
                        title={item.title}
                        items={item.items}
                        onItemClick={handleItemClick}
                        onItemHover={(hoveredItem, event) => handlePreviewChange(hoveredItem, event)}
                    />
                ))}
            </List>

            {/* Preview window that appears when hovering */}
            {previewItem && (
                <MediaPreview
                    item={previewItem}
                    position={mousePosition}
                    isFrozen={isPreviewFrozen}
                    onClose={onClose}
                />
            )}
        </div>
    );
};

function ResultCardTitle({doi, score, pdf_link=null, ...args}) {
    const doifixed: String = formDOI(doi)

    return (
    <Typography {...args} sx={{ display: 'flex', flexDirection: 'row' }}>
        <Link href={"https://arxiv.org/abs/" + doifixed} className='pr-4'> {formDOI(doi)}</Link>
        [
            <Link href={"https://arxiv.org/pdf/" + doifixed}>pdf</Link>,
            <Link href={"https://arxiv.org/format/" + doifixed}>other</Link>
        ]
    </Typography>
    );
}

// interface Result {
//     score: number,
//     id: string, //paper_id
//     title: string;
//     authors: string[];
//     abstract: string;
//     url: string;
// }

function ResultCard({index, result, ...args }) {
    const [hoveredItem, setHoveredItem] = useState(null);

    return (
        /* Result Card */
        <Card className='border-black border-5' {...args} >
            <Box className="flex">
                {/* Left Side */}
                <div className="flex-1 flex flex-col">
                    {/* Top title */}
                    <ResultCardTitle className="pl-4" index={index} doi={result.id} score={result.score} />

                    {/* Main Content */}
                    <div className="flex pl-4 pb-4">
                        {/* Article Information */}
                        <div className="flex-1 pr-4">
                            <Typography gutterBottom variant="h6" component="div">
                                { result.title }
                            </Typography>
                            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                { result.text }
                            </Typography>
                        </div>
                    </div>
                </div>

                {/* Right Side - Info */}
                <ResultItemInfo
                    className='flex-2 ml-5 max-w-70'
                    info={DUMMYrightPanelData}
                    // onPreviewChange={setHoveredItem}
                />
            </Box>
            <Typography className="justify-self-center pb-1" variant="body2" sx={{ color: 'text.secondary' }}>
                { "Faiss score: " + result.score.toFixed(3) }
            </Typography>
        </Card>

    );
}


export default function ResultArxivItem({index, result, ...args}) {
    // const authorText = result.authors ? result.authors.join(', ') : '';

    return (
        <ListItem sx={{ display: "list-item" }}>
            <ResultCard index={index} result={result} {...args}/>
        </ListItem>
    );
}
