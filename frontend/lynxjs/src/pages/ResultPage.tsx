import { useState } from '@lynx-js/react'
import { useParams } from 'react-router';
import { useLocation } from 'react-router';
import { ResultItem } from "./resultpage/ResultItem.tsx";
import { ResultSearchBar } from './resultpage/ResultSearchBar.tsx';


export function ResultPage() {
    let { query  } = useParams();
    const [onPage, setOnPage] = useState(query);
    const TAP = () => console.log(onPage);

    console.log("ResultPage", query);

    // const location = useLocation();
    // const queryParams = new URLSearchParams(location.search);
    // const asd = queryParams.get("q") || "";





    return (
        <view
            style={{
                width:"100%",
                height:"100%",
                //alignSelf:"center"
            }}
            bindtap={TAP}
        >
            {/* TOP BAR */}
            <ResultSearchBar />

            {/* Results */}
            <scroll-view style={{paddingTop:"5px", alignSelf:"center", justifySelf:"center"}}>
                {Array.from({ length: 20 }).map((item, index) => (
                    <ResultItem width="100vh" height="25vh" index={index} />
                ))}
            </scroll-view>
        </view>
    );
};