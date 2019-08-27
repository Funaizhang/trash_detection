local M = { }

function M.parse(arg)

    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------
    cmd:option('-imgColorPath',    '0000_color.png'   ,    'RGB Image path')
    cmd:option('-imgDepthPath',    '0000_depth.png'   ,    'Depth Image path')
    cmd:option('-resultPath',    '0000_results.h5'   ,    'Result hdf5 path')
    cmd:option('-outputScale',    1/8                ,    'Output scaling')
    cmd:option('-imgHeight',    424                 ,    'input image height')
    cmd:option('-imgWidth',    512                 ,    'input image width')
    cmd:text()

    local opt = cmd:parse(arg or {})
    return opt
end

return M
