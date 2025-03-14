import './SearchBar.css'
import searchIcon from "./search-icon.svg";
import { Button } from './Button.tsx';
import { useCallback, useEffect, useState, useLynxGlobalEventListener } from '@lynx-js/react'

export function ResultView({state, results}) {
    //const [isActive, setIsActive] = useState(false);

    return (
        <view className={`accordion-item ${state.active ? "active" : ""}`}>

        </view>
    );
};