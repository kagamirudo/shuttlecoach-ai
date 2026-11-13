import argparse
import json
from pathlib import Path

from .media import download, probe, write_json
from yt_dlp.utils import DownloadError


def main():
    parser = argparse.ArgumentParser(description="ShuttleCoach — local singles and shuttle tracking with gated reports")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest")
    ingest.add_argument("url")
    inspect = commands.add_parser("probe")
    inspect.add_argument("video")
    inspect.add_argument("--out")
    run = commands.add_parser("analyze")
    run.add_argument("config")
    run.add_argument("--model", default="models/yolo11m-pose.pt")
    run.add_argument("--device", choices=["auto","cpu","mps"], default="auto")
    validate = commands.add_parser("validate")
    validate.add_argument("run")
    validate.add_argument("--references", required=True)
    validate.add_argument("--checks", required=True)
    shuttle=commands.add_parser("shuttle")
    shuttle.add_argument("run")
    shuttle.add_argument("--device",choices=["auto","cpu","mps"],default="auto")
    shuttle.add_argument("--quality",action="store_true",help="Use overlapping temporal ensemble at the model dataset's 30 fps cadence")
    shuttle.add_argument('--candidates',action='store_true',help='Keep all TrackNet components in wider court airspace for player-aware association')
    shuttle.add_argument("--new-court-review",action="store_true",help="Create a separate raised-zone review, preserving prior shuttle outputs")
    shuttle_validate=commands.add_parser("validate-shuttle")
    shuttle_validate.add_argument("run")
    shuttle_validate.add_argument("--references",required=True)
    coaching=commands.add_parser("report")
    coaching.add_argument("run")
    recovery=commands.add_parser('recover-players');recovery.add_argument('run');recovery.add_argument('--device',choices=['auto','cpu','mps'],default='auto')
    associate=commands.add_parser('associate-shuttle');associate.add_argument('run');associate.add_argument('--review',required=True)
    motion=commands.add_parser('motion-preview');motion.add_argument('run')
    footwork=commands.add_parser('footwork');footwork.add_argument('run')
    llm=commands.add_parser('llm-report');llm.add_argument('run')
    args = parser.parse_args()
    try:
        if args.command == "ingest":
            print(download(args.url))
        elif args.command == "probe":
            metadata = probe(args.video)
            if args.out:
                write_json(Path(args.out), metadata)
            print(json.dumps({k:v for k,v in metadata.items() if k != "times"}, indent=2))
        elif args.command == "analyze":
            from .runner import analyze
            analyze(args.config, args.model, args.device)
        elif args.command == 'recover-players':
            from .player_recovery import recover_players
            recover_players(args.run,args.device)
        elif args.command=='associate-shuttle':
            from .shuttle_association import associate_shuttle
            associate_shuttle(args.run,args.review)
        elif args.command=='motion-preview':
            from .motion import write_motion_preview
            print(json.dumps(write_motion_preview(args.run),indent=2))
        elif args.command=='footwork':
            from .footwork import write_footwork
            result=write_footwork(args.run)
            print(json.dumps({'players':result['summaries'],'review_candidates':len(result['events'])},indent=2))
        elif args.command=='llm-report':
            from .llm_report import generate_llm_report
            result=generate_llm_report(args.run)
            print(f"Generated AI analysis ({result['model']}) for {len(result['analyses'])} players. View AI analysis in the app's Review & validate tab.")
        elif args.command == "shuttle":
            from .shuttle import track_shuttle
            if args.new_court_review:
                from .runner import revise_run
                parent=json.loads((Path(args.run)/"analysis.json").read_text())
                args.run=str(revise_run(args.run,parent["config"]["rallies"],reuse_birdie=False))
                print(f"Court-zone review: {args.run}",flush=True)
            if args.quality or args.candidates:
                path=Path(args.run)/"analysis.json";manifest=json.loads(path.read_text())
                if (path.parent/'shuttle.json').exists():
                    raise ValueError('This run already has shuttle results. Use --new-court-review --quality to preserve the prior run.')
                manifest['config']['shuttle_quality']=True;write_json(path,manifest)
                if args.candidates:
                    manifest['config']['shuttle_candidates']=True;write_json(path,manifest)
            try:
                track_shuttle(args.run,args.device)
            except BaseException as error:
                status_file=Path(args.run)/"shuttle_status.json"
                previous=json.loads(status_file.read_text()) if status_file.exists() else {}
                if status_file.parent.is_dir() and previous.get("state")!="complete":
                    write_json(status_file,{"state":"failed","error":str(error)})
                raise
        elif args.command == "validate-shuttle":
            from .shuttle import validate_shuttle
            result=validate_shuttle(args.run,args.references)
            print(json.dumps(result,indent=2))
            if result["status"]!="passed":raise SystemExit(2)
        elif args.command == "report":
            from .coaching import write_coaching_report
            print(json.dumps(write_coaching_report(args.run),indent=2))
        else:
            from .runner import validate_run
            result = validate_run(args.run, args.references, args.checks)
            print(json.dumps(result, indent=2))
            if result["status"] != "passed":
                raise SystemExit(2)
    except (ValueError, OSError, KeyError, RuntimeError, DownloadError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
