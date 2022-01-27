import click
from configparser import ConfigParser as _ConfigParser
from pathlib import Path


class ConfigParser(_ConfigParser):

    def read(self, paths):
        # TODO: type conversion
        super(ConfigParser, self).read(paths)
        return self


def custom_command(config_parser=ConfigParser):

    class CommandWithOverride(click.Command):

        def invoke(self, ctx):
            config_files = ctx.params.get("config_file", None)
            if config_files is not None:
                parser = config_parser().read(config_files)
                for section in parser.sections():
                    for param, value in parser[section].items():
                        if param in ctx.params and ctx.params[param] is None:
                            ctx.params[param] = value

            return super(CommandWithOverride, self).invoke(ctx)

    return CommandWithOverride


@click.command(cls=custom_command())
@click.argument("arg")
@click.option("--opt", type=int)
@click.option("-c", "--config_file", multiple=True, type=click.Path())
def main(arg, opt, config_file):
    print("arg: {}".format(arg))
    print("opt: {}".format(opt))
    import ipdb; ipdb.set_trace()
    print("config_file: {}".format(config_file))


main('my_arg --opt 1 -c default.cfg -c config.cfg'.split())
