import { useEffect, useState } from '@lynx-js/react'
import "./Button.css"

export function Button({ ontap=()=>{} }: { ontap: () => void }) {
    return (
        <view class='button' bindtap={ontap}>
            <text style={{
                alignSelf: "center",
                justifySelf: "center",
                position: "relative"
            }}>Search</text>
        </view>
    );
};