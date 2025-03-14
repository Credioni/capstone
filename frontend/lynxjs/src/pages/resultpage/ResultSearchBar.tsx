import { useState } from '@lynx-js/react'
import searchIcon from "./search-icon.svg";


export function ResultSearchBar({searchBarValue, setSearchBarValue, onsubmit}: any) {
    const onSubmit = () => onsubmit(searchBarValue);
    const handleInput = (e: any) => {
        setSearchBarValue(e.detail.value);
    };

    return (
        <view style={{
            justifyContent:"center",
            alignContent:"center",
            width: "100%",
            height: "5vh",
            backgroundColor: "#00000070"
        }}>
            <text style={{
                position:"relative",
                justifySelf:"center",
                alignSelf:"center"
            }}>Search Bar</text>
        </view>
    );
};