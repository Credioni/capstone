import { Card, CardHeader, ListItem } from "@mui/material";
import React, { useState } from "react";

function ResultItemVideo({index=1, video, ...args}) {
  const [videoId, setVideoId] = useState(video.video_id);

  return (
        <Card className="w-full" {...args}>
            <CardHeader className="w-full pl-4">
                {video.title}
            </CardHeader>
            <div className="max-w-lg">
                <iframe
                    width="100%"
                    height="350"
                    src={`https://www.youtube.com/embed/${videoId}`}
                    title="YouTube video player"
                    frameBorder="0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                    allowFullScreen
                ></iframe>
            </div>
        </Card>
    );
};

function ResultListItemVideo({video, ...args}) {
    // const authorText = result.authors ? result.authors.join(', ') : '';

    return (
        <ListItem className="w-full" sx={{ display: "list-item" }}>
            <ResultItemVideo video={video} {...args}/>
        </ListItem>
    );
}

export default ResultListItemVideo;
