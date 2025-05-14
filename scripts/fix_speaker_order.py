#!/usr/bin/env python3
"""
Utility to analyze and fix speaker labels in transcripts.
This can help ensure consistent speaker numbers across multiple files.
"""
import os
import re
import glob
from pathlib import Path
import argparse

def analyze_speakers(file_path):
    """Analyze speakers in a transcript file and return statistics."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all speaker mentions
    speaker_pattern = r'Speaker (\d+)'
    speakers = re.findall(speaker_pattern, content)
    
    # Convert to integers and get unique speakers
    speaker_nums = [int(s) for s in speakers]
    unique_speakers = sorted(set(speaker_nums))
    
    # Count speaker occurrences
    speaker_counts = {f"Speaker {num}": speaker_nums.count(num) for num in unique_speakers}
    
    return {
        'total_speakers': len(unique_speakers),
        'speaker_ids': unique_speakers,
        'speaker_counts': speaker_counts,
        'total_segments': len(speakers),
        'content': content
    }

def fix_speaker_labels(file_path, verbose=True):
    """Fix speaker labels in a transcript file to ensure they start from 1 and are sequential."""
    stats = analyze_speakers(file_path)
    content = stats['content']
    speaker_ids = stats['speaker_ids']
    
    # If speakers are already sequential starting from 1, no need to fix
    if speaker_ids == list(range(1, len(speaker_ids) + 1)):
        if verbose:
            print(f"  No fix needed for {file_path}, speakers already sequential")
        return False
    
    # Create a mapping from old speaker IDs to sequential ones
    speaker_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(speaker_ids), 1)}
    
    # Replace all occurrences using regex
    for old_id, new_id in speaker_mapping.items():
        if old_id != new_id:
            # Use a regex pattern with word boundaries to avoid partial matches
            pattern = rf'\bSpeaker {old_id}\b'
            content = re.sub(pattern, f'Speaker {new_id}', content)
    
    # Write the fixed content back to the file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    if verbose:
        print(f"  Fixed speaker labels in {file_path}: {speaker_mapping}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Analyze and fix speaker diarization in transcripts")
    parser.add_argument('--dir', type=str, default='./output/transcripts', help="Directory containing transcripts")
    parser.add_argument('--fix', action='store_true', help="Fix speaker numbers to be sequential starting from 1")
    parser.add_argument('--file', type=str, help="Process a specific file instead of a directory")
    parser.add_argument('--verbose', action='store_true', help="Print detailed information")
    args = parser.parse_args()
    
    if args.file:
        # Process a single file
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File {file_path} not found")
            return
            
        print(f"Processing file: {file_path}")
        stats = analyze_speakers(file_path)
        print(f"  Total speakers: {stats['total_speakers']}")
        print(f"  Speaker IDs: {stats['speaker_ids']}")
        print(f"  Speaker segments: {stats['speaker_counts']}")
        print(f"  Total segments: {stats['total_segments']}")
        
        if args.fix:
            fixed = fix_speaker_labels(file_path, args.verbose)
            if fixed:
                print(f"  Fixed speaker labels in {file_path}")
            else:
                print(f"  No fixes needed for {file_path}")
        
    else:
        # Process all files in directory
        transcript_dir = Path(args.dir)
        transcript_files = list(transcript_dir.glob('**/*.srt')) + list(transcript_dir.glob('**/*.txt'))
        
        if not transcript_files:
            print(f"No transcript files found in {args.dir}")
            return
        
        print(f"Found {len(transcript_files)} transcript files")
        
        # Analyze each file
        fixed_count = 0
        for file_path in transcript_files:
            stats = analyze_speakers(file_path)
            
            if args.verbose:
                print(f"\nFile: {file_path}")
                print(f"  Total speakers: {stats['total_speakers']}")
                print(f"  Speaker IDs: {stats['speaker_ids']}")
                print(f"  Speaker segments: {stats['speaker_counts']}")
                print(f"  Total segments: {stats['total_segments']}")
            
            # Fix speaker labels if requested
            if args.fix:
                if fix_speaker_labels(file_path, False):
                    fixed_count += 1
                    print(f"Fixed speaker labels in {file_path}")
        
        if args.fix:
            print(f"\nFixed speaker labels in {fixed_count} out of {len(transcript_files)} files")
        else:
            print("\nRun with --fix to fix speaker labels")

if __name__ == "__main__":
    main()
