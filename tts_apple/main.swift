//
//  main.swift
//  tts
//
//  Created by Muhammad Hilmi Asyrofi on 21/1/20.
//  Copyright © 2020 Muhammad Hilmi Asyrofi. All rights reserved.
//
//  Text-to-Speech on OSX

import Foundation
import AppKit
import Cocoa

// File path (change this).
let path = "/Users/mhilmiasyrofi/Documents/test-case-generation/corpus-sentence.txt"
// Read an entire text file into an NSString.
let contents = try NSString(contentsOfFile: path,
    encoding: String.Encoding.ascii.rawValue)

let synth = NSSpeechSynthesizer()

let voices = NSSpeechSynthesizer.availableVoices.map { v in (v, NSSpeechSynthesizer.attributes(forVoice: v)[NSSpeechSynthesizer.VoiceAttributeKey.localeIdentifier] as! String) }.sorted  { ($0.1, $0.0.rawValue) < ($1.1, $1.0.rawValue) }
for (k, v) in voices {
    if (v.contains("en")) {
        if (v.contains("US") && k.rawValue.contains("Alex")) {
            let voice_id = k.rawValue
            // let accent_id = v
            // let directory = String("/Users/mhilmiasyrofi/Documents/test-case-generation/tts_apple/aiff_generated_speech/")

            synth.setVoice(NSSpeechSynthesizer.VoiceName(rawValue: voice_id))

            // Process each lines
            var i = 0
            contents.enumerateLines({ (line, stop) -> () in
                i = i + 1
                let output_file =  String(format:"/Users/mhilmiasyrofi/Documents/test-case-generation/tts_apple/aiff_generated_speech/audio_%d.aiff", i)
                
                let url = Foundation.URL.init(fileURLWithPath: output_file)

                let skill_executor = ""
                let skill_command = line
                let command = skill_executor + skill_command
                synth.startSpeaking(command, to: url)
                print("Generated: " + command)
            })
        }
    }
}




