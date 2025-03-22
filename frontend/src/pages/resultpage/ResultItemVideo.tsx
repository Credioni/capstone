import React, { useState } from "react";
import { Box, Card, CardContent, CardHeader, ListItem, Tooltip, Typography } from "@mui/material";


function ResultItemVideo({index=1, video, ...args}) {
    const [videoId, setVideoId] = useState(video.video_id);

    return (
        <Card className="w-fit" {...args}>
            <Box className="h-24 flex flex-col justify-center">
                <Typography className="w-full pl-4" variant="body1">
                    {"From "} <b>{video.author}</b>
                </Typography>
                <Typography
                    className="w-full pl-4 overflow-hidden"
                    variant="body1"
                    sx={{
                        display: '-webkit-box',
                        WebkitLineClamp: 2,
                        WebkitBoxOrient: 'vertical',
                        height: '48px'
                    }}
                >
                    <i>{video.title}</i>
                </Typography>
            </Box>
            <CardContent sx={{ maxWidth: '100%', padding: '16px', minHeight: "300px", minWidth:"400px" }}>
                <iframe
                    width="400"
                    height="300"
                    src={`https://www.youtube.com/embed/${videoId}`}
                    title="YouTube video player"
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                ></iframe>
            </CardContent>
            <Typography className="justify-self-center" variant="body2">
                {"Faiss score: " + video.score.toFixed(3) }
            </Typography>
        </Card>
    );
}

function ResultListItemVideo({video, ...args}) {
    // const authorText = result.authors ? result.authors.join(', ') : '';
    return (
        <ListItem className="w-full">
            <ResultItemVideo video={video} {...args}/>
        </ListItem>
    );
}

export default ResultListItemVideo;
