import { Typography } from '@mui/material';
import React, { useState, useEffect } from 'react';

const LoadingText = ({list}) => {
    const [selectedItem, setSelectedItem] = useState(list[0]);
    const [index, setIndex] = useState(0);

    useEffect(() => {
        // Set an interval to change the selected item every second
        const intervalId = setInterval(() => {
            setIndex((prevIndex) => (prevIndex + 1) % list.length);
        }, 5000);

        return () => clearInterval(intervalId);
    }, [list.length]);

    // Update the selected item based on the current index
    useEffect(() => {
        setSelectedItem(list[index]);
    }, [index]);

    return (
        <Typography variant='h5'>
            { selectedItem }
        </Typography>
    );
};

export default LoadingText;