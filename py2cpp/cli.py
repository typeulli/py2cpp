from pathlib import Path

import click

from py2cpp.core.compiler import Setting, py_2_cpp, py_2_exe

@click.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('-o', '--output', 'output_path', required=False, type=click.Path(), help='Path to the output executable/file.')
@click.option('-c', '--compile', 'compile_target', required=False, type=click.Choice(['auto', 'cpp', 'exe']), default='auto', help='Target compilation format.')
@click.option('-p', '--print', 'print_code', is_flag=True, help='Print the generated C++ code to stdout instead of writing to a file (only for cpp target).')
def main(input_path: str, output_path: str | None, compile_target: str, print_code: bool):
    setting = Setting(minimize_namespace=["string", "vector", "cout", "cin", "endl", "get"])

    input_path_obj = Path(input_path)
    python_code = input_path_obj.read_text(encoding="utf-8")
    
    if print_code:
        print(py_2_cpp(python_code, path=str(input_path_obj), setting=setting))
        return
    
    output_path_obj = Path(output_path) if output_path else input_path_obj.with_suffix("." + compile_target)

    if compile_target == "auto":
        if output_path_obj.suffix == ".exe":
            compile_target = "exe"
        else:
            compile_target = "cpp"

    match compile_target or "cpp":
        case 'cpp':
            try:
                cpp_code = py_2_cpp(python_code, path=str(input_path_obj), setting=setting).code
            except Exception as e:
                raise click.ClickException(f"Error converting Python to C++: {e}")
            output_path_obj.write_text(cpp_code, encoding="utf-8")
        case 'exe':
            py_2_exe(python_code, path=str(output_path_obj), verbose=True, setting=setting)
        case _:
            raise click.UsageError("Invalid compile target specified: " + str(compile_target))