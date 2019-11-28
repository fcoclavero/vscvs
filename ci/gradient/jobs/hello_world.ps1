gradient jobs create `
--name hello_world `
--optionsFile config.yaml `
--command ( `
    'echo hello_world $$ echo path && echo $PATH && echo env && printenv && ' + `
    'echo pip freeze && python --version && pip freeze'
)
