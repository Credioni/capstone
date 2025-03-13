import './SearchBar.css'
import searchIcon from "./search-icon.svg";
import { DragAndDrop } from '../../common/DragNDrop.tsx';
import { useCallback, useEffect, useState, useLynxGlobalEventListener } from '@lynx-js/react'

export function SearchBar() {

    return (
        <view style={{
            alignItems: "center",
            justifyContent: "center",
            width: "100%",
            marginTop: "20px",
        }}>

            {/* Search container with rounded borders */}
            <view style={{
                display: "flex",
                flexDirection: "row",
                alignItems: "center",
                width: "95%",
                padding: "10px 16px",
                borderRadius: "50px",
                border: "1px solid #ccc",
                boxShadow: "0 2px 5px rgba(0,0,0,0.1)",
            }}>
                {/* Magnifying glass icon */}
                <image
                    src={searchIcon}
                    style={{
                        width: "20px",
                        height: "20px",
                        marginRight: "10px",
                        opacity: 0.7,
                    }}
                />

                {/* Input field without its own border */}
                <input
                    type="text"
                    placeholder="Search..."
                    user-select="none"
                    style={{
                        flex: 1,
                        fontSize: "1rem",
                        padding: "2px 0",
                        border: "none",
                        outline: "none",
                        background: "transparent",
                    }}
                />
            </view>
            {/* Search button */}
            <view
                //bindtap={handleKeyPress}
                style={{
                    marginTop: "2%",
                    //marginLeft: "10px",
                    //paddingTop: "10px",
                    height: "3.5vh",
                    width: "35%",
                    backgroundColor: "#4A90E2", // Blue button
                    color: "white",
                    border: "none",
                    borderRadius: "25px",
                    fontSize: "0.9rem",
                    fontWeight: "500",
                    cursor: "pointer",
                    transition: "background-color 0.2s ease",
                }}
            >
                <text style={{alignSelf: "center", justifySelf: "center", position: "relative"}}>Search</text>
            </view>
        </view>
    );
};