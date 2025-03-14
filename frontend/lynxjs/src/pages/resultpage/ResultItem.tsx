import { useState } from '@lynx-js/react'
import searchIcon from "./search-icon.svg";


export function ResultItem(props: { width: string; height: string; index: number }) {
    const angle = 90 + (props.index | 1);

    return (
        <view style={{
            width: props.width,
            height: props.height,
            background: `linear-gradient(${angle}deg, rgba(255, 53, 26, 0.25), rgba(0, 235, 235, 0.25))`,
            marginBottom:"10px",

            borderWidth:"2px",
            borderRadius:"10px",
            borderStyle: "solid",
            borderColor:"black",
        }}>
            <text style={{
                position:"relative",
                justifySelf:"center",
                alignSelf:"center"
            }}>Result</text>
        </view>
    );
};