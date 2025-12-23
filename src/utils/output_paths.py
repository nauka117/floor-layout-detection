from pathlib import Path
from typing import Optional


def prepare_output_path(
    source_name: str,
    output_target: Optional[Path],
    is_single: bool,
    default_suffix: str,
) -> Optional[Path]:

    if output_target is None:
        return None
    
    if is_single and output_target.suffix:
        return output_target
    
    
    output_dir = output_target if not output_target.suffix else output_target.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{source_name}_result{default_suffix}"
