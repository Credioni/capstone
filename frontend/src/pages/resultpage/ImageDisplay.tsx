import React from 'react';
import {
  Card,
  CardMedia,
  CardContent,
  Typography,
  Box,
} from '@mui/material';

const ImageDisplay = ({ images, ...args }) => {
    // If no images are provided, return nothing
    if (!images || images.length === 0) {
        return null;
    }

    return (
        <Box {...args} className="flex" component="div" style={{ display: 'flex', flexWrap: 'wrap', gap: '16px' }}>
            {images.map((image, index) => (
                <Card key={index} sx={{ width: 240, height: 320, display: 'flex', flexDirection: 'column' }}>
                    <CardContent sx={{ padding: 1 }}>
                        <Typography variant="h6" noWrap>
                            {image.id ? `${image.id.split('.')[0]}` : `Image ${index + 1}`}
                        </Typography>
                    </CardContent>

                    <CardMedia
                        component="img"
                        sx={{
                            height: 200,
                            objectFit: 'contain',
                            padding: 1
                        }}
                        image={`data:image/png;base64,${image.base64}`}
                        alt={`Image ${image.id || index + 1}`}
                    />

                    <CardContent sx={{ mt: 'auto', padding: 2, paddingTop: 1 }}>
                        <Typography variant="body2" align="center">
                            Faiss Score: {image.score.toFixed(3)}
                        </Typography>
                    </CardContent>
                </Card>
            ))}
        </Box>
    );
};

export default ImageDisplay;