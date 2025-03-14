import { useState } from '@lynx-js/react'
import './SearchBar.css'
import searchIcon from "./search-icon.svg";
import { Button } from './Button.tsx';

export function SearchBar({onsubmit}: any) {
    const onSubmit = () => onsubmit(inputValue);

    const [inputValue, setInputValue] = useState("");

    const handleInput = (e: any) => {
        setInputValue(e.detail.value);
    };

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
                    value={inputValue}
                    /* @ts-ignore */
                    bindinput={handleInput}
                    type="text"
                    placeholder="Search..."
                    inputMode='text'
                    //user-select="none"

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
            <Button ontap={onSubmit}/>
        </view>
    );
};